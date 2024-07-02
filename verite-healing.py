import pandas as pd

df = pd.read_csv('data/VERITE/VERITE_with_evidence.csv')

def update_paths(path):
    new_path = path.replace('/data/VERITE/external_evidence', 'data/VERITE/external_evidence')
    return new_path

df['image_path'] = df['image_path'].apply(update_paths)
df['images_paths'] = df['images_paths'].apply(lambda x: str([update_paths(p) for p in eval(x)]))

df.to_csv('data/VERITE/VERITE_with_evidence.csv', index=False)

print("Paths have been updated and the file has been saved as 'updated_csv_file.csv'")

input_file = 'data/VERITE/VERITE_ranked_evidence_clip_ViTL14.csv'
output_file = 'data/VERITE/VERITE_ranked_evidence_clip_ViTL14.csv'

columns_to_keep = [
    'caption', 'image_path', 'captions', 'len_text_info',
    'images_paths', 'num_images', 'label',
    'img_ranked_items', 'img_sim_scores',
    'txt_ranked_items', 'txt_sim_scores'
]

df = pd.read_csv(input_file)

df = df[columns_to_keep]

df.to_csv(output_file)

print(f"Modified CSV file has been saved to {output_file}")
