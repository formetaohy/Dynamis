use font8x8::UnicodeFonts;

pub const CELL_WIDTH: u32 = 8;
pub const CELL_HEIGHT: u32 = 10;
pub const SCALE: u32 = 2;
pub const TEXTURE_WIDTH: u32 = 2048;
pub const TEXTURE_HEIGHT: u32 = 256;
pub const TEXT_ORIGIN: (f32, f32) = (12.0, 12.0);

pub struct Raster {
    pub pixels: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

pub fn raster(text: &str) -> Raster {
    let lines: Vec<&str> = text.split('\n').collect();
    let columns = lines
        .iter()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0);
    assert!(
        columns as u32 * CELL_WIDTH * SCALE <= TEXTURE_WIDTH,
        "hud line exceeds texture width"
    );
    assert!(
        lines.len() as u32 * CELL_HEIGHT * SCALE <= TEXTURE_HEIGHT,
        "hud exceeds texture height"
    );
    let width = columns as u32 * CELL_WIDTH * SCALE;
    let height = lines.len() as u32 * CELL_HEIGHT * SCALE;
    let mut pixels = vec![0u8; (TEXTURE_WIDTH * height) as usize];
    for (row_index, line) in lines.iter().enumerate() {
        for (column_index, character) in line.chars().enumerate() {
            let glyph = font8x8::BASIC_FONTS
                .get(character)
                .unwrap_or_else(|| panic!("hud contains unsupported character {character:?}"));
            for (glyph_y, bits) in glyph.iter().enumerate() {
                for glyph_x in 0..8 {
                    if bits & (1 << glyph_x) == 0 {
                        continue;
                    }
                    for scale_y in 0..SCALE {
                        for scale_x in 0..SCALE {
                            let pixel_x = column_index as u32 * CELL_WIDTH * SCALE
                                + glyph_x as u32 * SCALE
                                + scale_x;
                            let pixel_y = row_index as u32 * CELL_HEIGHT * SCALE
                                + glyph_y as u32 * SCALE
                                + scale_y;
                            pixels[(pixel_y * TEXTURE_WIDTH + pixel_x) as usize] = 255;
                        }
                    }
                }
            }
        }
    }
    Raster {
        pixels,
        width,
        height,
    }
}
