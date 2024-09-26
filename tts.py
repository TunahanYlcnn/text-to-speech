import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio
import os
import sys


def metni_parcalara_ayir(metin, max_harf=225):
    parcalar = []
    baslangic = 0
    metin_uzunlugu = len(metin)
    noktalama_isaretleri = {'.', ',', ';', '!', '?'}

    while baslangic < metin_uzunlugu:
        son = baslangic + max_harf

        # Eğer son, metin uzunluğunu aşarsa, sonu metnin uzunluğuna eşitle
        if son >= metin_uzunlugu:
            son = metin_uzunlugu
            parcalar.append(metin[baslangic:son].strip())
            break

        # Maksimum karakter uzunluğunu aşmadan önce bir noktalama işareti bulmaya çalış
        while son > baslangic and metin[son] not in noktalama_isaretleri:
            son -= 1

        # Eğer bir noktalama işareti bulunmazsa, maximum harf sayısında kes
        if son == baslangic:
            son = baslangic + max_harf
        else:
            # Noktalama işaretinden sonra başlasın diye sonu 1 artırıyoruz
            son += 1

        # Parçayı ekle
        parcalar.append(metin[baslangic:son].strip())

        # Bir sonraki parça için başlangıcı ayarla
        baslangic = son

    return parcalar


def siiri_duz_metin_yap(siir):
    # Satırları böl ve aralarındaki gereksiz boşlukları temizle
    satirlar = siir.splitlines()
    duz_metin = ""

    for i, satir in enumerate(satirlar):
        satir = satir.strip()
        if not satir:  # Boş satırları atla
            continue

        # Eğer son karakter nokta veya virgül değilse ve bu satır son değilse, virgül ekle
        if i < len(satirlar) - 1 and satir[-1] not in ['.', ',']:
            satir += ','

        duz_metin += satir + " "

    return duz_metin.strip()


def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    config = XttsConfig()
    config.load_json(xtts_config)

    model = Xtts.init_from_config(config)
    print("XTTS modeli yükleniyor...")
    model.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)

    if torch.cuda.is_available():
        model.cuda()

    print("Model başarıyla yüklendi!")
    return model


def run_tts(model, tts_text, speaker_audio_file):
    if model is None or not speaker_audio_file:
        raise ValueError("Model veya referans ses dosyası yüklü değil!")

    # GPT koşullandırma latenti ve konuşmacı embedding'ini al
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=speaker_audio_file,
        gpt_cond_len=model.config.gpt_cond_len,
        max_ref_length=model.config.max_ref_len,
        sound_norm_refs=model.config.sound_norm_refs
    )

    # Tek bir inference çağrısı ile tüm metni işle
    out = model.inference(
        text=tts_text,
        language='tr',
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=model.config.temperature,
        length_penalty=model.config.length_penalty,
        repetition_penalty=model.config.repetition_penalty,
        top_k=model.config.top_k,
        top_p=model.config.top_p
    )

    # Tüm çıktıyı tek bir tensor olarak döndür
    audio_tensor = torch.tensor(out['wav']).unsqueeze(0)
    return audio_tensor


def save_wav_file(wav_tensor, output_path):
    torchaudio.save(output_path, wav_tensor, 24000)
    print(f"Ses dosyası başarıyla kaydedildi: {output_path}")


def main():
    i = 1

    xtts_checkpoint = r'C:\Users\zeyne\Desktop\bark\AtatürkEğitilmişModel3\best_model.pth'
    xtts_config = r'C:\Users\zeyne\Desktop\bark\AtatürkEğitilmişModel3\config (1).json'
    xtts_vocab = r'C:\Users\zeyne\Desktop\bark\AtatürkEğitilmişModel3\vocab (1).json'
    speaker_audio_file = r'C:\Users\zeyne\Desktop\bark\ATATÜRK\0902 (1).MP3'

    model = load_model(xtts_checkpoint, xtts_config, xtts_vocab)

    while True:
        # Kullanıcıdan çok satırlı input olarak şiiri al
        print("Lütfen bir metin giriniz (Ctrl+D ile sonlandırabilirsiniz):")
metinler = []
        while True:
            try:
                satir = input()
                if not satir.strip():  # Boş bir satır algılandığında döngüden çık
                    break
                metinler.append(satir)
            except EOFError:  # Ctrl+D tuşuna basıldığında EOFError yakalanır
                if metinler:
                    break
                else:
                    print("Boş metin girildi. Çıkılıyor...")
                    return

        text = "\n".join(metinler).strip()

        if not text:
            print("Boş metin girildi. Çıkılıyor...")
            break

        # Şiiri düz metin haline getir
        duz_metin = siiri_duz_metin_yap(text)

        # Metni parçalara ayıralım
        parcalar = metni_parcalara_ayir(duz_metin)

        # Tüm parçaları içeren bir liste oluşturalım
        combined_wav = []
        try:
            # Her bir parça için ses dosyasını oluştur ve birleştir
            for parca in parcalar:
                print(f"İşleniyor: {parca[:50]}...")  # Parçanın başından bir kesit
                wav_tensor = run_tts(model, parca, speaker_audio_file)
                if wav_tensor is not None:
                    combined_wav.append(wav_tensor)
                else:
                    print("Ses dosyası oluşturulamadı.")
        except Exception as e:
            print(f"Döngü başarısız: {e}")

        if not combined_wav:
            print("Birleştirilecek ses dosyası bulunamadı.")
            continue

        # Tüm parçaları birleştir
        combined_wav_tensor = torch.cat(combined_wav, dim=1)

        # Dosya yolunu belirtiyoruz
        output_path = f'C:\\Users\\zeyne\\Desktop\\bark\\output_audio\\genclige_hitabe{i}.wav'

        # Birleştirilmiş ses dosyasını kaydet
        save_wav_file(combined_wav_tensor, output_path)
        i += 1

if name == "main":
    main()