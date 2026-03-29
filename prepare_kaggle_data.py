import os
import shutil
import kaggle
import csv
import random


DATASET_NAME = "paramaggarwal/fashion-product-images-small"
RAW_DIR = "kaggle_raw"
FINAL_DIR = "dataset"
MAX_IMAGES_PER_CATEGORY = 80

CSV_MAPPING = {
    "watch": ["Watches"],
    "sunglasses": ["Eyewear"],
    "handbag": ["Bags", "Wallets"],
    "hat": ["Headwear"],

    "high heels": ["Heels", "Sandals"],
    "sneakers": ["Shoes"],
    "running shoes": ["Flip Flops", "Sandals"],

    "jeans": ["Jeans"],
    "t-shirt": ["Topwear", "Tshirts"],
    "jacket": ["Jackets", "Coats", "Blazers"],
    "dress": ["Dress", "One Piece"]
}


def setup_guaranteed_dataset():
    print(f"Stahuji dataset {DATASET_NAME}...")
    # Tento dataset má složku 'images' a soubor 'styles.csv'
    if not os.path.exists(RAW_DIR):
        kaggle.api.dataset_download_files(DATASET_NAME, path=RAW_DIR, unzip=True)
    print("Staženo.")

    if os.path.exists(FINAL_DIR):
        shutil.rmtree(FINAL_DIR)

    # Vytvoříme 11 složek
    my_categories = ["watch", "sunglasses", "handbag", "hat", "high heels",
                     "sneakers", "running shoes", "jeans", "t-shirt", "jacket", "dress"]

    for cat in my_categories:
        os.makedirs(os.path.join(FINAL_DIR, cat), exist_ok=True)

    print("Čtu styles.csv a třídím obrázky...")

    # Cesta k CSV a obrázkům
    csv_path = os.path.join(RAW_DIR, "styles.csv")
    images_dir = os.path.join(RAW_DIR, "images")

    # Slovník pro sběr ID obrázků: {"watch": [1111.jpg, 2222.jpg], ...}
    found_files = {cat: [] for cat in my_categories}

    # Čtení CSV (pokud existuje)
    if not os.path.exists(csv_path):
        print("CHYBA: styles.csv nenalezen! Něco je špatně s datasetem.")
        return

    # Otevřeme CSV a projdeme řádek po řádku
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Získáme info o produktu
            img_id = row['id'] + ".jpg"
            master_cat = row['masterCategory']  # Apparel, Accessories...
            sub_cat = row['subCategory']  # Watches, Shoes...
            article_type = row['articleType']  # Jeans, Heels, Jackets...


            # Zkoušíme to napasovat do  kategorií
            target = None

            # 1. Hodinky (Watch)
            if sub_cat == "Watches":
                target = "watch"

            # 2. Brýle (Sunglasses)
            elif sub_cat == "Eyewear" and "Sunglasses" in article_type:
                target = "sunglasses"

            # 3. Klobouky (Hat)
            elif sub_cat == "Headwear":
                target = "hat"

            # 4. Kabelky (Handbag)
            elif sub_cat == "Bags" or "Handbags" in article_type:
                target = "handbag"

            # 5. Podpatky (High heels)
            elif "Heels" in article_type:
                target = "high heels"

            # 6. Tenisky (Sneakers) vs Běžecké (Running)
            elif sub_cat == "Shoes":
                if "Sports" in row['usage']:
                    target = "running shoes"  # Sportovní boty
                elif "Casual" in row['usage']:
                    target = "sneakers"  # Volnočasové

            # 7. Oblečení
            elif article_type == "Jeans":
                target = "jeans"
            elif article_type in ["Jackets", "Coats", "Blazers"]:
                target = "jacket"
            elif article_type in ["Dresses", "Jumpsuit"]:
                target = "dress"
            elif article_type in ["Tshirts", "Tops", "Shirts"]:
                target = "t-shirt"

            # Pokud jsme našli kategorii a obrázek existuje, přidáme ho do seznamu
            if target and target in found_files:
                src_path = os.path.join(images_dir, img_id)
                if os.path.exists(src_path):
                    found_files[target].append(src_path)

    # Nyní kopírujeme náhodný výběr do tvých složek
    print("Kopíruji soubory...")
    for cat, file_list in found_files.items():
        if not file_list:
            print(f"Pro kategorii '{cat}' jsem nenašel žádné fotky")
            continue

        count = min(len(file_list), MAX_IMAGES_PER_CATEGORY)
        to_copy = random.sample(file_list, count)

        for src in to_copy:
            dst = os.path.join(FINAL_DIR, cat, os.path.basename(src))
            shutil.copy(src, dst)

        print(f"{cat}: Připraveno {count} obrázků.")


    shutil.rmtree(RAW_DIR)
    print("Složka 'dataset' je plná a připravená.")


if __name__ == "__main__":
    setup_guaranteed_dataset()