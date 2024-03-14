import argparse
import os

from tokenizers import AddedToken
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import AutoTokenizer

sample = {
    "ar": "دعني أعتني بهذا. الرجل الذي اشترى ممتلكاتي يعيش في المدينة.",
    "de": "Dafür lassen Sie mich sorgen.--Der Mann, der mein Gut gekauft hat, wohnt in der Stadt.",
    "el": "Αφήστε με να το φροντίσω. Ο άνθρωπος που αγόρασε την περιουσία μου μένει στην πόλη.",
    "en": "Let me take care of that. The man who bought my property lives in the city.",
    "en2": "Let me take care of that. The man who bought my property lives in the city.",
    "es": "Déjame encargarme de eso. El hombre que compró mi propiedad vive en la ciudad.",
    "fi": "Hieno, tuskin huomattava hymyily kiertyi Sinikka-rouvan kapeiden huulten ympärille.",
    "ko": "그건 내가 처리하겠습니다. 내 재산을 산 사람은 도시에 살아요.",
    "nl": "Laat mij daarvoor zorgen. De man die mijn eigendom heeft gekocht, woont in de stad.",
    "pl": "Pozwól mi się tym zająć. Człowiek, który kupił moją nieruchomość, mieszka w mieście.",
    "ru": "Позвольте мне позаботиться об этом. Мужчина, который купил мою недвижимость, живет в городе.",
    "tr": "Bırakın bununla ben ilgileneyim. Benim mülkümü satın alan adam şehirde yaşıyor.",
    "ar_en": "Let me take care of that. The man who bought my property lives in the city."
             "دعني أعتني بهذا. الرجل الذي اشترى ممتلكاتي يعيش في المدينة.",
    "de_en": "Dafür lassen Sie mich sorgen.--Der Mann, der mein Gut gekauft hat, wohnt in der Stadt."
             "Let me take care of that. The man who bought my property lives in the city.",
    "el_en": "Αφήστε με να το φροντίσω. Ο άνθρωπος που αγόρασε την περιουσία μου μένει στην πόλη."
             "Let me take care of that. The man who bought my property lives in the city.",
    "es_en": "Déjame encargarme de eso. El hombre que compró mi propiedad vive en la ciudad."
             "Let me take care of that. The man who bought my property lives in the city.",
    "fi_en": "Hieno, tuskin huomattava hymyily kiertyi Sinikka-rouvan kapeiden huulten ympärille."
             "Let me take care of that. The man who bought my property lives in the city.",
    "ko_en": "그건 내가 처리하겠습니다. 내 재산을 산 사람은 도시에 살아요."
             "Let me take care of that. The man who bought my property lives in the city.",
    "nl_en": "Laat mij daarvoor zorgen. De man die mijn eigendom heeft gekocht, woont in de stad."
             "Let me take care of that. The man who bought my property lives in the city.",
    "pl_en": "Pozwól mi się tym zająć. Człowiek, który kupił moją nieruchomość, mieszka w mieście."
             "Let me take care of that. The man who bought my property lives in the city.",
    "ru_en": "Позвольте мне позаботиться об этом. Мужчина, который купил мою недвижимость, живет в городе."
             "Let me take care of that. The man who bought my property lives in the city.",
    "tr_en": "Bırakın bununla ben ilgileneyim. Benim mülkümü satın alan adam şehirde yaşıyor."
             "Let me take care of that. The man who bought my property lives in the city.",
}


def train_tokenizer(model, dataset, lang):

    bpe_tokenizer = ByteLevelBPETokenizer()

    files = [f"./data/{dataset}/{lang}/raw/{split}.txt" for split in ["test", "train", "validation"]]

    bpe_tokenizer.train(files=files, vocab_size=32000, min_frequency=2)

    tokenizer_path = f'./data/{dataset}/{lang}/{model}_tokenizer'
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    # save the vocab.json and merges.txt files of the trained bpe tokenizer
    bpe_tokenizer.save_model(tokenizer_path)

    model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, tokenizer_type=model)
    model_tokenizer.model_max_length = 512
    model_tokenizer.add_special_tokens({"pad_token": AddedToken("<pad>", normalized=True)})

    print(f'Tokenizer vocab size: {len(model_tokenizer)}')
    print(f'Tokenizer max sequence length: {model_tokenizer.model_max_length} \n')

    # save the full model tokenizer configuration files
    model_tokenizer.save_pretrained(tokenizer_path)

    if lang in sample:
        output = model_tokenizer.encode_plus(sample[lang])
        print(output.tokens(), '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="the tokenizer configuration to use")
    parser.add_argument("dataset", type=str, help="the dataset to be processed")
    parser.add_argument("language", type=str, help="the language of the files")
    args = parser.parse_args()

    print(f'\nTraining a custom {args.model} tokenizer for data {args.dataset}_{args.language}\n')
    train_tokenizer(args.model, args.dataset, args.language)
