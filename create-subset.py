import json
import os
import shutil
import pandas as pd

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file)

def load_and_merge_data(data_path, dataset_type, vn_data):
    dataset = load_json(f"{data_path}/news_clippings/data/news_clippings/data/merged_balanced/{dataset_type}.json")
    dataset_df = pd.DataFrame(dataset["annotations"])
    dataset_df.insert(0, 'new_clipping_id', range(len(dataset_df)))
    dataset_df.columns.values[1] = 'article_id'

    merged_data = pd.merge(dataset_df, vn_data, left_on='article_id', right_on='id', how='left')
    merged_data = merged_data.rename(columns={'image_path': 'article_id_image_path', 'article_path': 'article_id_article_path'})

    final_merged_data = pd.merge(merged_data, vn_data, left_on='image_id', right_on='id', how='left')
    final_merged_data = final_merged_data.rename(columns={'image_path': 'image_id_image_path', 'article_path': 'image_id_article_path'})
    
    return final_merged_data

def update_annotations(data_path, last_id):
    data = load_json(f"{data_path}/news_clippings/data/news_clippings/data/merged_balanced/train.json")
    data['annotations'] = data['annotations'][:last_id + 1]
    save_json(data, f"{data_path}/news_clippings/data/news_clippings/data/merged_balanced/train.json")

def filter_visualnews_data(data_path, ids_to_keep):
    vn_data = load_json(f'{data_path}/VisualNews/origin/data.json')
    ids_to_keep_set = set(ids_to_keep)

    filtered_data = []
    base_image_path = f"{data_path}/VisualNews/origin"
    
    for item in vn_data:
        if item['id'] in ids_to_keep_set:
            filtered_data.append(item)
        else:
            image_path = item.get('image_path')
            article_path = item.get('article_path')

            if image_path:
                full_image_path = os.path.join(base_image_path, image_path.lstrip('./'))
                if os.path.isfile(full_image_path):
                    try:
                        os.remove(full_image_path)
                    except OSError as e:
                        print(f"Error deleting file {full_image_path}: {e}")

            if article_path and os.path.isfile(article_path):
                try:
                    os.remove(article_path)
                except OSError as e:
                    print(f"Error deleting file {article_path}: {e}")

    save_json(filtered_data, f'{data_path}/VisualNews/origin/data.json')

def clean_queries_dataset(data_path, last_id):
    data = load_json(f'{data_path}/news_clippings/queries_dataset/dataset_items_train.json')
    new_data = {}

    base_directory_path = f"{data_path}/news_clippings/queries_dataset/merged_balanced"

    for key, value in data.items():
        key_int = int(key)
        if key_int <= last_id:
            new_data[key] = value
        else:
            if 'inv_path' in value:
                full_inv_path = os.path.join(base_directory_path, value['inv_path'].lstrip('./'))
                if os.path.exists(full_inv_path):
                    try:
                        shutil.rmtree(full_inv_path)
                    except OSError as e:
                        print(f"Error deleting directory {full_inv_path}: {e}")

            if 'direct_path' in value:
                full_direct_path = os.path.join(base_directory_path, value['direct_path'].lstrip('./'))
                if os.path.exists(full_direct_path):
                    try:
                        shutil.rmtree(full_direct_path)
                    except OSError as e:
                        print(f"Error deleting directory {full_direct_path}: {e}")

    save_json(new_data, f'{data_path}/news_clippings/queries_dataset/dataset_items_train.json')

def main():
    data_path = 'data/'
    SOURCE_EVIDENCE_PATH = f'{data_path}/news_clippings/queries_dataset'

    train_data = load_json(f"{data_path}news_clippings/data/news_clippings/data/merged_balanced/train.json")
    train_df = pd.DataFrame(train_data["annotations"])
    train_df.insert(0, 'new_clipping_id', range(len(train_df)))
    train_df.columns.values[1] = 'article_id'

    vn_data = pd.DataFrame(load_json(f'{data_path}/VisualNews/origin/data.json'))[['id', 'image_path', 'article_path']]

    train_paths = pd.DataFrame(load_json(f'{SOURCE_EVIDENCE_PATH}/dataset_items_train.json')).transpose().reset_index().rename(columns={'index': 'match_index'})
    train_paths['match_index'] = train_paths['match_index'].astype(int)

    merged_train_data = pd.merge(train_df, train_paths, left_on='new_clipping_id', right_on='match_index')
    merged_with_article_data = pd.merge(merged_train_data, vn_data, left_on='article_id', right_on='id', how='left')
    merged_with_article_data = merged_with_article_data.rename(columns={'image_path': 'article_id_image_path', 'article_path': 'article_id_article_path'})
    
    final_merged_data = pd.merge(merged_with_article_data, vn_data, left_on='image_id', right_on='id', how='left')
    final_merged_data = final_merged_data.rename(columns={'image_path': 'image_id_image_path', 'article_path': 'image_id_article_path'})

    num_entries_to_keep = len(final_merged_data) // 10
    subset_final_merged_data = final_merged_data.head(num_entries_to_keep)
    last_new_clipping_id = subset_final_merged_data['new_clipping_id'].max()

    update_annotations(data_path, last_new_clipping_id)

    val_data = load_and_merge_data(data_path, 'val', vn_data)
    test_data = load_and_merge_data(data_path, 'test', vn_data)

    all_ids = pd.concat([
        val_data['id_x'], val_data['id_y'],
        test_data['id_x'], test_data['id_y'],
        subset_final_merged_data['id_x'], subset_final_merged_data['id_y']
    ]).unique()

    filter_visualnews_data(data_path, all_ids)

    clean_queries_dataset(data_path, last_new_clipping_id)

if __name__ == "__main__":
    main()
