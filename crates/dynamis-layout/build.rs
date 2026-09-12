use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const ABI: &str = "abi/rigid.wgsl";
const CONSTANTS: &str = "src/constant.rs";
const OUTPUT: &str = "records.rs";
const RECORD_SUFFIX: &str = "Record";

#[derive(Clone, Copy, Debug)]
struct Measure {
    size: u32,
    align: u32,
}

#[derive(Clone, Debug)]
enum Count {
    Literal(u32),
    Named(String),
}

#[derive(Clone, Debug)]
enum Ty {
    F32,
    U32,
    Atomic,
    Vec2,
    Vec3,
    Vec4,
    Array(Box<Ty>, Count),
    Record(String),
}

#[derive(Clone, Debug)]
struct Record {
    name: String,
    fields: Vec<(String, Ty)>,
}

struct Abi {
    records: BTreeMap<String, Record>,
    constants: BTreeMap<String, u32>,
    layouts: BTreeMap<String, Measure>,
}

impl Abi {
    fn load() -> Self {
        let mut abi = Self {
            records: parse_records(&fs::read_to_string(ABI).expect("the abi source is readable")),
            constants: parse_constants(
                &fs::read_to_string(CONSTANTS).expect("the constants source is readable"),
            ),
            layouts: BTreeMap::new(),
        };
        let names = abi.records.keys().cloned().collect::<Vec<_>>();
        for name in names {
            abi.measure_record(&name, &mut Vec::new());
        }
        abi
    }

    fn measure_record(&mut self, name: &str, resolving: &mut Vec<String>) -> Measure {
        if let Some(measured) = self.layouts.get(name) {
            return *measured;
        }
        assert!(
            !resolving.iter().any(|pending| pending == name),
            "struct {name} embeds itself"
        );
        resolving.push(name.to_owned());
        let fields = self.records[name].fields.clone();
        let mut end = 0;
        let mut align = 1;
        for (field, ty) in &fields {
            let measured = self.measure_field(ty, resolving, name, field);
            end = round_up(end, measured.align) + measured.size;
            align = align.max(measured.align);
        }
        resolving.pop();
        let layout = Measure {
            size: round_up(end, align),
            align,
        };
        self.layouts.insert(name.to_owned(), layout);
        layout
    }

    fn measure_field(
        &mut self,
        ty: &Ty,
        resolving: &mut Vec<String>,
        owner: &str,
        field: &str,
    ) -> Measure {
        match ty {
            Ty::F32 | Ty::U32 | Ty::Atomic => Measure { size: 4, align: 4 },
            Ty::Vec2 => Measure { size: 8, align: 8 },
            Ty::Vec3 => Measure {
                size: 12,
                align: 16,
            },
            Ty::Vec4 => Measure {
                size: 16,
                align: 16,
            },
            Ty::Record(name) => self.measure_record(name, resolving),
            Ty::Array(element, count) => {
                let measured = self.measure_field(element, resolving, owner, field);
                assert!(
                    measured.size == round_up(measured.size, measured.align),
                    "field {field} of struct {owner} arrays a type whose storage stride exceeds its rust width"
                );
                Measure {
                    size: measured.size * self.count(count, owner, field),
                    align: measured.align,
                }
            }
        }
    }

    fn measure(&self, ty: &Ty) -> Measure {
        match ty {
            Ty::F32 | Ty::U32 | Ty::Atomic => Measure { size: 4, align: 4 },
            Ty::Vec2 => Measure { size: 8, align: 8 },
            Ty::Vec3 => Measure {
                size: 12,
                align: 16,
            },
            Ty::Vec4 => Measure {
                size: 16,
                align: 16,
            },
            Ty::Record(name) => self.layouts[name],
            Ty::Array(element, count) => {
                let measured = self.measure(element);
                Measure {
                    size: measured.size * self.count_value(count),
                    align: measured.align,
                }
            }
        }
    }

    fn count(&self, count: &Count, owner: &str, field: &str) -> u32 {
        match count {
            Count::Literal(literal) => *literal,
            Count::Named(name) => *self.constants.get(name).unwrap_or_else(|| {
                panic!("field {field} of struct {owner} names the undeclared constant {name}")
            }),
        }
    }

    fn count_value(&self, count: &Count) -> u32 {
        match count {
            Count::Literal(literal) => *literal,
            Count::Named(name) => self.constants[name],
        }
    }

    fn rust_size(&self, ty: &Ty) -> u32 {
        match ty {
            Ty::F32 | Ty::U32 | Ty::Atomic => 4,
            Ty::Vec2 => 8,
            Ty::Vec3 => 12,
            Ty::Vec4 => 16,
            Ty::Record(name) => self.layouts[name].size,
            Ty::Array(element, count) => self.rust_size(element) * self.count_value(count),
        }
    }

    fn rust_type(&self, ty: &Ty) -> String {
        match ty {
            Ty::F32 => "f32".to_owned(),
            Ty::U32 | Ty::Atomic => "u32".to_owned(),
            Ty::Vec2 => "[f32; 2]".to_owned(),
            Ty::Vec3 => "[f32; 3]".to_owned(),
            Ty::Vec4 => "[f32; 4]".to_owned(),
            Ty::Record(name) => format!("{name}{RECORD_SUFFIX}"),
            Ty::Array(element, count) => {
                let length = match count {
                    Count::Literal(literal) => literal.to_string(),
                    Count::Named(name) => format!("crate::constant::{name} as usize"),
                };
                format!("[{}; {length}]", self.rust_type(element))
            }
        }
    }

    fn rust_align(&self, ty: &Ty) -> u32 {
        match ty {
            Ty::F32 | Ty::U32 | Ty::Atomic | Ty::Vec2 | Ty::Vec3 | Ty::Vec4 => 4,
            Ty::Record(name) => self.records[name]
                .fields
                .iter()
                .map(|(_, field)| self.rust_align(field))
                .fold(1, u32::max),
            Ty::Array(element, _) => self.rust_align(element),
        }
    }
}

fn round_up(value: u32, align: u32) -> u32 {
    value.div_ceil(align) * align
}

fn parse_records(source: &str) -> BTreeMap<String, Record> {
    let tokens = tokenize(source);
    let mut records = BTreeMap::new();
    let mut cursor = 0;
    while cursor < tokens.len() {
        expect(&tokens, &mut cursor, "struct");
        let name = tokens[cursor].clone();
        cursor += 1;
        expect(&tokens, &mut cursor, "{");
        let mut fields = Vec::new();
        while tokens[cursor] != "}" {
            let field = tokens[cursor].clone();
            cursor += 1;
            expect(&tokens, &mut cursor, ":");
            fields.push((field, parse_type(&tokens, &mut cursor)));
            expect(&tokens, &mut cursor, ",");
        }
        expect(&tokens, &mut cursor, "}");
        let replaced = records.insert(name.clone(), Record { name, fields });
        assert!(replaced.is_none(), "the abi source repeats a struct");
    }
    records
}

fn parse_type(tokens: &[String], cursor: &mut usize) -> Ty {
    let token = tokens[*cursor].clone();
    *cursor += 1;
    match token.as_str() {
        "f32" => Ty::F32,
        "u32" => Ty::U32,
        "vec2f" => Ty::Vec2,
        "vec3f" => Ty::Vec3,
        "vec4f" => Ty::Vec4,
        "atomic" => {
            expect(tokens, cursor, "<");
            expect(tokens, cursor, "u32");
            expect(tokens, cursor, ">");
            Ty::Atomic
        }
        "array" => {
            expect(tokens, cursor, "<");
            let element = parse_type(tokens, cursor);
            expect(tokens, cursor, ",");
            let count = tokens[*cursor].clone();
            *cursor += 1;
            expect(tokens, cursor, ">");
            let count = match count.parse::<u32>() {
                Ok(literal) => Count::Literal(literal),
                Err(_) => Count::Named(count),
            };
            Ty::Array(Box::new(element), count)
        }
        name => Ty::Record(name.to_owned()),
    }
}

fn expect(tokens: &[String], cursor: &mut usize, expected: &str) {
    assert_eq!(
        tokens[*cursor], expected,
        "the abi source must write {expected:?}"
    );
    *cursor += 1;
}

fn tokenize(source: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = source.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '/' if chars.peek() == Some(&'/') => {
                for skipped in chars.by_ref() {
                    if skipped == '\n' {
                        break;
                    }
                }
            }
            '{' | '}' | '<' | '>' | ':' | ',' | ';' | '=' => {
                flush(&mut tokens, &mut current);
                tokens.push(character.to_string());
            }
            character if character.is_whitespace() => flush(&mut tokens, &mut current),
            character => current.push(character),
        }
    }
    flush(&mut tokens, &mut current);
    tokens
}

fn flush(tokens: &mut Vec<String>, current: &mut String) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

fn parse_constants(source: &str) -> BTreeMap<String, u32> {
    let tokens = tokenize(source);
    let mut constants = BTreeMap::new();
    let mut cursor = 0;
    while cursor + 5 < tokens.len() {
        if tokens[cursor] == "const" && tokens[cursor + 2] == ":" && tokens[cursor + 4] == "=" {
            if let Ok(value) = tokens[cursor + 5].parse::<u32>() {
                constants.insert(tokens[cursor + 1].clone(), value);
            }
            cursor += 6;
            continue;
        }
        cursor += 1;
    }
    constants
}

fn main() {
    println!("cargo:rerun-if-changed={ABI}");
    println!("cargo:rerun-if-changed={CONSTANTS}");
    let abi = Abi::load();
    let mut output = String::new();
    for record in abi.records.values() {
        emit_record(&mut output, &abi, record);
    }
    let path = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join(OUTPUT);
    fs::write(path, output).expect("the generated records are writable");
}

fn emit_record(output: &mut String, abi: &Abi, record: &Record) {
    let rust_name = format!("{}{RECORD_SUFFIX}", record.name);
    writeln!(output, "#[repr(C)]").unwrap();
    writeln!(
        output,
        "#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]"
    )
    .unwrap();
    writeln!(output, "pub struct {rust_name} {{").unwrap();

    let mut offsets = Vec::new();
    let mut wgsl_offset = 0;
    let mut rust_offset = 0;
    let mut pads = 0;
    for (field, ty) in &record.fields {
        let measured = abi.measure(ty);
        wgsl_offset = round_up(wgsl_offset, measured.align);
        rust_offset = round_up(rust_offset, abi.rust_align(ty));
        assert!(
            rust_offset <= wgsl_offset,
            "field {field} of struct {} outruns the shader layout",
            record.name
        );
        if rust_offset < wgsl_offset {
            let gap = wgsl_offset - rust_offset;
            writeln!(output, "    pub _wgsl_pad{pads}: [u8; {gap}],").unwrap();
            pads += 1;
            rust_offset += gap;
        }
        writeln!(output, "    pub {field}: {},", abi.rust_type(ty)).unwrap();
        offsets.push((field.clone(), wgsl_offset));
        wgsl_offset += measured.size;
        rust_offset += abi.rust_size(ty);
    }
    let size = abi.layouts[&record.name].size;
    if rust_offset < size {
        writeln!(
            output,
            "    pub _wgsl_pad{pads}: [u8; {}],",
            size - rust_offset
        )
        .unwrap();
    }
    writeln!(output, "}}").unwrap();

    writeln!(output, "const _: () = {{").unwrap();
    writeln!(
        output,
        "    assert!(core::mem::size_of::<{rust_name}>() == {size});"
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
}
