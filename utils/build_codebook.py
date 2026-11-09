import json
from collections import Counter

def create_codebook(input_json_path, output_json_path):
    print("Ham yamalar yükleniyor... Bu biraz sürebilir.")
    # 1. 865 bin yamanın olduğu json dosyanı oku
    with open(input_json_path, 'r') as f:
        all_patches = json.load(f)

    hash_counts = Counter()
    hash_to_template = {} # Her topolojiden sadece BİR örnek (şablon) tutacağız

    print("Yamalar analiz ediliyor ve şablonlar çıkarılıyor...")
    # 2. Tüm yamaları dön, frekansları say ve her hash için bir "Şablon" ayır
    for patch_id, patch_data in all_patches.items():
        w_hash = patch_data["wl_hash"]
        hash_counts[w_hash] += 1
        
        # Eğer bu WL-Hash'i ilk defa görüyorsak, onun lokal koordinatlarını ve 
        # element dizilişini "Lego Şablonu" olarak kaydediyoruz.
        if w_hash not in hash_to_template:
            hash_to_template[w_hash] = {
                "local_nodes": patch_data["local_nodes"],
                "local_elements": patch_data["local_elements"]
            }

    # 3. En çok tekrar edenden en aza doğru sırala (Rank 1'den Rank 29'a)
    sorted_hashes = hash_counts.most_common()

    # 4. Nihai Sözlüğü (Codebook) İnşa Et
    codebook = {}

    # Önce Yapay Zeka için Özel Tokenları (0, 1, 2) Ekliyoruz
    codebook["0"] = {"type": "SPECIAL", "name": "<PAD>", "description": "Padding (Doldurma)"}
    codebook["1"] = {"type": "SPECIAL", "name": "<START>", "description": "Diziyi Başlat"}
    codebook["2"] = {"type": "SPECIAL", "name": "<END>", "description": "Diziyi Bitir (Mesh Tamamlandı)"}
    codebook["3"] = {"type": "SPECIAL", "name": "<UNK>", "description": "Bilinmeyen Patch"}


    # Şimdi Gerçek 29 Yamayı Token 4'ten itibaren ekliyoruz
    token_id = 4
    for w_hash, count in sorted_hashes:
        template = hash_to_template[w_hash]
        
        codebook[str(token_id)] = {
            "type": "PATCH",
            "wl_hash": w_hash,
            "frequency": count,
            "local_nodes": template["local_nodes"],
            "local_elements": template["local_elements"]
        }
        token_id += 1

    # 5. JSON olarak kaydet
    with open(output_json_path, "w") as f:
        json.dump(codebook, f, indent=4)

    print(f"\nBAŞARILI! Codebook oluşturuldu ve '{output_json_path}' olarak kaydedildi.")
    print(f"Toplam Token Sayısı: {len(codebook)} (3 Özel + {len(sorted_hashes)} Yama)")

# Kodu Çalıştır
# 'extracted_patches.json' yerine kendi büyük veri dosyanın adını yaz
create_codebook('extracted_patches.json', 'codebook.json')
