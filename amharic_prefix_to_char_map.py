# Script to map file name prefixes to actual Amharic characters arranged in a 34 x 7 grid

# Example mapping dictionary: prefix -> Amharic character
# Replace the placeholders with actual Amharic characters as needed

prefix_to_char = {
    '001he': 'ሀ', '002hu': 'ሁ', '003hi': 'ሂ', '004ha': 'ሃ', '005he2': 'ሄ', '006hə': 'ህ', '007ho': 'ሆ',
    '008le': 'ለ', '009lu': 'ሉ', '010li': 'ሊ', '011la': 'ላ', '012le2': 'ሌ', '013lə': 'ል', '014lo': 'ሎ',
    # ... continue for all 34 x 7 = 238 mappings
}

def print_mapping_grid(mapping, rows=34, cols=7):
    keys = list(mapping.keys())
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            idx = r * cols + c
            if idx < len(keys):
                prefix = keys[idx]
                char = mapping[prefix]
                row_chars.append(f"{prefix}:{char}")
            else:
                row_chars.append("N/A")
        print("\t".join(row_chars))

if __name__ == "__main__":
    print("Amharic Prefix to Character Mapping (34 x 7 grid):")
    print_mapping_grid(prefix_to_char)
