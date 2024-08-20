from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import copy
import nltk

def load_translation(weight_path, model_name):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    state = torch.load(weight_path, map_location=torch.device('cpu'))
    model.load_state_dict(state, strict=False)
    return model

class Fragment:
    placeHolders = ["Germany", "Rome", "Finland", "Canada", "Australia", "Africa", "Europe"]
    translation = ""

    def __init__(self, str: str, lang: str, subFrag: list = [], matLang: bool = False, num: int = -1):
        self.str = str
        self.lang = lang
        self.subFrag = subFrag
        self.matLang = matLang
        self.num = num

    def __repr__(self):
        s = ""
        if not self.subFrag:
            s = self.str
        for frag in self.subFrag:
            if frag.lang == "placeHolder":
                s = s + " [" + str(frag.num) + "]"
            else:
                s = s + " " + frag.str
        return s + " (" + self.lang + ")"

    def __add__(self, other):
        if self.lang == "" or other.lang == "":
            return Fragment(self.str + " " + other.str, self.lang + other.lang, self.subFrag + other.subFrag, False, 0)
        elif self.lang == "" and other.lang == "":
            raise Exception("Neither Fragment has a language")
        elif self.lang != other.lang and self.lang != "placeHolder" and other.lang != "placeHolder":
            raise Exception("Fragments must have the same language")
        else:
            if self.lang == "placeHolder":
                lang = other.lang
            elif other.lang == "placeHolder":
                lang = self.lang
            else:
                lang = self.lang
            subFrag = []
            for f in (self, other):
                if not f.subFrag:
                    subFrag = subFrag + [f]
                else:
                    subFrag = subFrag + f.subFrag
            return Fragment(self.str + " " + other.str, lang, subFrag, self.matLang)

    def replace(self, newFrag):
        holderNum = newFrag.num
        frag = Fragment.empty()
        for subFrag in self.subFrag:
            if subFrag.num == holderNum:
                frag = frag + Fragment(newFrag.translation, "English")
            else:
                frag = frag + subFrag
        return frag

    def swapToTranslation(self):
        f = copy.deepcopy(self)
        f.str = ""
        f.lang = "English"
        f.subFrag = []
        for word in f.translation.split():
            distance = []
            for hold in f.placeHolders:
                distance.append(nltk.edit_distance(hold, word))
            if min(distance) <= 1:
                idx = distance.index(min(distance))
                frag = Fragment(f.placeHolders[idx], "placeHolder", [], True, idx)
            else:
                frag = Fragment(word, "English")
            f = f + frag
        f.translation = self.str
        return f

    @classmethod
    def empty(cls):
        return cls("", "")

class FullModel:
    def __init__(self, model, tokenizer):
        self.outputLayer = ['English', 'Luganda']
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, input):
        output = self.tokenizer(input, return_tensors="pt")
        return output

    def decode(self, input):
        output = self.tokenizer.batch_decode(input, skip_special_tokens=True)
        return output

    def generate(self, input):
        output = self.model.generate(**input)
        return output

    def compute(self, input):
        encodedInput = self.encode(input)
        output = self.generate(encodedInput)
        decodedOutput = self.decode(output)
        return decodedOutput

class Pipeline:
    placeHolders = Fragment.placeHolders

    def __init__(self, transModels: dict):
        self.transModels = transModels

    def translate(self, input: str, input_lang: str, output_lang: str):
        if input_lang == output_lang:
            return input
        return self.transModels[input_lang].compute(input)[0]

def Create_Pipeline(input_lang):
    # Determine the appropriate model and tokenizer based on input language
    if input_lang == "English":
        model_name = "Helsinki-NLP/opus-mt-en-mul"
    else:
        model_name = "Helsinki-NLP/opus-mt-mul-en"
    
    # Load Translation tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define the paths for the translation models
    translationPaths = {
        "English": "E:/MODELs/Eng2Lug(Multi).pth",  # English to Luganda
        "Luganda": "E:/MODELs/Lug2Eng(Multi).pth"   # Luganda to English
    }

    # Ensure that the correct model is loaded based on the input language
    translationModels = {input_lang: FullModel(load_translation(translationPaths[input_lang], model_name), tokenizer)}
    pipeline = Pipeline(translationModels)
    return pipeline

def run():
    print("Select the translation model:")
    print("1. English to Luganda")
    print("2. Luganda to English")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        input_lang = "English"
        output_lang = "Luganda"
    elif choice == "2":
        input_lang = "Luganda"
        output_lang = "English"
    else:
        print("Invalid choice. Exiting.")
        return

    inputSentence = input(f"Enter the sentence to translate from {input_lang} to {output_lang}: ")
    pipeline = Create_Pipeline(input_lang)
    output = pipeline.translate(inputSentence, input_lang, output_lang)

    print("\nTranslated Sentence:\n")
    print(output)

if __name__ == "__main__":
    run()
