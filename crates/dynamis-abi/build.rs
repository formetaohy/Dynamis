use naga::front::wgsl::parse_str;
use naga::{
    ArraySize, Constant, Expression, Handle, Literal, Module, Scalar, ScalarKind, StructMember,
    Type, TypeInner,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const RECORDS: &str = "abi/records.wgsl";
const OUTPUT: &str = "records.rs";
const LENGTHS: &str = "lengths.rs";
const RECORD_SUFFIX: &str = "Record";

struct Abi {
    module: Module,
}

struct Record {
    members: Vec<StructMember>,
    span: u32,
}

fn main() {
    println!("cargo:rerun-if-changed={RECORDS}");
    let abi = Abi::load();
    let mut output = String::new();
    for (name, record) in abi.records() {
        abi.emit_record(&mut output, &name, &record);
    }
    let lengths = abi.lengths();
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    fs::write(out.join(OUTPUT), output).expect("the generated records are writable");
    fs::write(out.join(LENGTHS), lengths).expect("the generated lengths are writable");
}

impl Abi {
    fn load() -> Self {
        let source = fs::read_to_string(RECORDS).expect("the abi source is readable");
        let module = parse_str(&source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string_with_path(&source, RECORDS)));
        Self { module }
    }

    fn lengths(&self) -> String {
        let mut lengths = BTreeMap::new();
        for (_, constant) in self.module.constants.iter() {
            let name = constant
                .name
                .clone()
                .unwrap_or_else(|| panic!("the abi source declares an unnamed constant"));
            lengths.insert(name, self.literal(constant));
        }
        let mut output = String::new();
        for (name, value) in lengths {
            writeln!(output, "pub const {name}: u32 = {value};").unwrap();
        }
        output
    }

    fn literal(&self, constant: &Constant) -> u32 {
        match self.module.global_expressions[constant.init] {
            Expression::Literal(Literal::U32(value)) => value,
            ref other => panic!("a constant of the abi source does not fold to a u32: {other:?}"),
        }
    }

    fn records(&self) -> BTreeMap<String, Record> {
        let mut records = BTreeMap::new();
        for (_, ty) in self.module.types.iter() {
            let TypeInner::Struct { members, span } = &ty.inner else {
                continue;
            };
            let name = ty
                .name
                .clone()
                .unwrap_or_else(|| panic!("the abi source declares an unnamed struct"));
            records.insert(
                name,
                Record {
                    members: members.clone(),
                    span: *span,
                },
            );
        }
        records
    }

    fn emit_record(&self, output: &mut String, name: &str, record: &Record) {
        let rust_name = format!("{name}{RECORD_SUFFIX}");
        writeln!(output, "#[repr(C)]").unwrap();
        writeln!(
            output,
            "#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]"
        )
        .unwrap();
        writeln!(output, "pub struct {rust_name} {{").unwrap();

        let mut offsets = Vec::new();
        let mut rust_offset = 0;
        let mut pads = 0;
        for member in &record.members {
            let field = member
                .name
                .as_deref()
                .unwrap_or_else(|| panic!("a record of the abi source declares an unnamed field"));
            rust_offset = round_up(rust_offset, self.rust_align(member.ty));
            assert!(
                rust_offset <= member.offset,
                "field {field} of struct {name} outruns the shader layout"
            );
            if rust_offset < member.offset {
                let gap = member.offset - rust_offset;
                writeln!(output, "    pub _wgsl_pad{pads}: [u8; {gap}],").unwrap();
                pads += 1;
                rust_offset += gap;
            }
            writeln!(output, "    pub {field}: {},", self.rust_type(member.ty)).unwrap();
            offsets.push((field, member.offset));
            rust_offset += self.rust_size(member.ty);
        }
        if rust_offset < record.span {
            writeln!(
                output,
                "    pub _wgsl_pad{pads}: [u8; {}],",
                record.span - rust_offset
            )
            .unwrap();
        }
        writeln!(output, "}}").unwrap();

        writeln!(output, "const _: () = {{").unwrap();
        writeln!(
            output,
            "    assert!(core::mem::size_of::<{rust_name}>() == {});",
            record.span
        )
        .unwrap();
        for (field, offset) in &offsets {
            writeln!(
                output,
                "    assert!(core::mem::offset_of!({rust_name}, {field}) == {offset});"
            )
            .unwrap();
        }
        writeln!(output, "}};").unwrap();
        writeln!(
            output,
            "impl crate::StreamRecord for {rust_name} {{ const WGSL: &'static str = {name:?}; }}"
        )
        .unwrap();
    }

    fn rust_type(&self, handle: Handle<Type>) -> String {
        match &self.module.types[handle].inner {
            TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => {
                self.scalar_type(scalar).to_owned()
            }
            TypeInner::Vector { size, scalar } => {
                format!("[{}; {}]", self.scalar_type(scalar), u32::from(*size))
            }
            TypeInner::Array { base, .. } => {
                format!("[{}; {}]", self.rust_type(*base), self.array_length(handle))
            }
            TypeInner::Struct { .. } => format!("{}{RECORD_SUFFIX}", self.record_name(handle)),
            other => panic!("the abi source uses a type the generator does not map: {other:?}"),
        }
    }

    fn record_name(&self, handle: Handle<Type>) -> &str {
        self.module.types[handle]
            .name
            .as_deref()
            .unwrap_or_else(|| panic!("the abi source declares an unnamed record"))
    }

    fn scalar_type(&self, scalar: &Scalar) -> &'static str {
        match (scalar.kind, scalar.width) {
            (ScalarKind::Float, 4) => "f32",
            (ScalarKind::Uint, 4) => "u32",
            _ => panic!("the abi source uses a scalar the generator does not map: {scalar:?}"),
        }
    }

    fn array_length(&self, handle: Handle<Type>) -> u32 {
        match self.module.types[handle].inner {
            TypeInner::Array {
                size: ArraySize::Constant(length),
                ..
            } => length.get(),
            ref other => {
                panic!("the abi source arrays a type the generator cannot bound: {other:?}")
            }
        }
    }

    fn rust_size(&self, handle: Handle<Type>) -> u32 {
        match &self.module.types[handle].inner {
            TypeInner::Scalar(_) | TypeInner::Atomic(_) => 4,
            TypeInner::Vector { size, .. } => u32::from(*size) * 4,
            TypeInner::Array { base, stride, .. } => {
                let element = self.rust_size(*base);
                assert!(
                    element == *stride,
                    "the abi source arrays a type of width {element} whose stride is {stride}"
                );
                element * self.array_length(handle)
            }
            TypeInner::Struct { span, .. } => *span,
            other => panic!("the abi source uses a type the generator does not map: {other:?}"),
        }
    }

    fn rust_align(&self, handle: Handle<Type>) -> u32 {
        match &self.module.types[handle].inner {
            TypeInner::Scalar(_) | TypeInner::Atomic(_) | TypeInner::Vector { .. } => 4,
            TypeInner::Array { base, .. } => self.rust_align(*base),
            TypeInner::Struct { members, .. } => members
                .iter()
                .map(|member| self.rust_align(member.ty))
                .max()
                .unwrap_or(1),
            other => panic!("the abi source uses a type the generator does not map: {other:?}"),
        }
    }
}

fn round_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}
