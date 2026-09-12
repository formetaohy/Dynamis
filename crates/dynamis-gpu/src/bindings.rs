use crate::BindingKind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
    pub name: String,
}

pub fn parse_bindings(source: &str) -> Vec<ShaderBinding> {
    let mut bindings = Vec::new();
    let mut cursor = 0;
    while let Some(offset) = source[cursor..].find("@group(") {
        cursor += offset + "@group(".len();
        let group = parse_index(source, &mut cursor, "group");
        expect(source, &mut cursor, ")");
        skip_space(source, &mut cursor);
        expect(source, &mut cursor, "@binding(");
        let binding = parse_index(source, &mut cursor, "binding");
        expect(source, &mut cursor, ")");
        skip_space(source, &mut cursor);
        expect(source, &mut cursor, "var<");
        let access = read_until(source, &mut cursor, '>');
        skip_space(source, &mut cursor);
        let declaration = read_until(source, &mut cursor, ':');
        let name = declaration.trim().to_owned();
        assert!(!name.is_empty(), "every shader binding needs a name");
        assert!(
            !bindings
                .iter()
                .any(|entry: &ShaderBinding| entry.group == group && entry.binding == binding),
            "the shader declares binding {binding} of group {group} twice"
        );
        assert!(
            !bindings
                .iter()
                .any(|entry: &ShaderBinding| entry.group == group && entry.name == name),
            "the shader declares the binding {name:?} of group {group} twice"
        );
        bindings.push(ShaderBinding {
            group,
            binding,
            kind: kind_of(&access),
            name,
        });
    }
    bindings
}

fn kind_of(access: &str) -> BindingKind {
    let fields = access
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    match fields[..] {
        ["uniform"] => BindingKind::Uniform,
        ["storage", "read"] => BindingKind::ReadOnlyStorage,
        ["storage", "read_write"] => BindingKind::ReadWriteStorage,
        _ => panic!("the shader declares an unknown binding access {access:?}"),
    }
}

fn parse_index(source: &str, cursor: &mut usize, what: &str) -> u32 {
    let start = *cursor;
    while source
        .as_bytes()
        .get(*cursor)
        .is_some_and(u8::is_ascii_digit)
    {
        *cursor += 1;
    }
    source[start..*cursor]
        .parse()
        .unwrap_or_else(|_| panic!("every shader binding needs a numeric {what} index"))
}

fn read_until(source: &str, cursor: &mut usize, terminator: char) -> String {
    let start = *cursor;
    let end = source[start..]
        .find(terminator)
        .unwrap_or_else(|| panic!("the shader binding misses a {terminator:?}"));
    *cursor = start + end + terminator.len_utf8();
    source[start..start + end].to_owned()
}

fn expect(source: &str, cursor: &mut usize, expected: &str) {
    assert!(
        source[*cursor..].starts_with(expected),
        "the shader binding must declare {expected:?}"
    );
    *cursor += expected.len();
}

fn skip_space(source: &str, cursor: &mut usize) {
    while source[*cursor..].starts_with(char::is_whitespace) {
        *cursor += 1;
    }
}
