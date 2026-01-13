import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MALE_NAMES = [
    "John", "Paul", "Mike", "Kevin", "Steve", "Greg", "Jeff", "Bill", "Peter", "Jack",
    "Ryan", "Thomas", "Brian", "Adam", "Harry", "Frank", "Edward", "Jerry", "George", "Arthur",
    "Jacob", "Tyler", "Aaron", "Henry", "Douglas", "Roger", "Walter", "Albert", "Harold", "Jonathan",
    "Justin", "Terry", "Gerald", "Keith", "Samuel", "Willie", "Ralph", "Lawrence", "Nicholas", "Roy",
    "Benjamin", "Bruce", "Brandon", "Donald", "Scott", "Dennis", "Patrick", "Alexander", "Raymond", "Gregory"
]

FEMALE_NAMES = [
    "Mary", "Jennifer", "Lisa", "Michelle", "Sarah", "Kim", "Amy", "Susan", "Rebecca", "Anna",
    "Emily", "Elizabeth", "Alice", "Amanda", "Laura", "Cynthia", "Linda", "Jessica", "Patricia", "Barbara",
    "Angela", "Christine", "Debra", "Rachel", "Janet", "Catherine", "Maria", "Heather", "Diane", "Virginia",
    "Julie", "Joyce", "Victoria", "Kelly", "Lauren", "Martha", "Judith", "Cheryl", "Megan", "Andrea",
    "Ann", "Evelyn", "Jean", "Kathryn", "Jacqueline", "Hannah", "Carol", "Gloria", "Teresa", "Sara"
]

EA_NAMES = ["Greg", "Emily", "Anne", "Jill", "Todd", "Neil", "Geoffrey", "Brett", "Brendan", "Laurie"]
AA_NAMES = ["Lakisha", "Jamal", "Darnell", "Tyrone", "Latoya", "Ebony", "Keisha", "Jermaine", "Tanisha", "DeShawn"]

def load_model():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=False
    ).to(DEVICE)
    model.eval()
    return model, tokenizer

def get_embedding(model, tokenizer, text, layer_idx=-1):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()

def get_dataset_embeddings(model, tokenizer, names, prompt_template, layer_idx=-1):
    embeddings = []
    for name in names:
        text = f"<|system|>\n{prompt_template}<|end|>\n<|user|>\n{name}<|end|>\n<|assistant|>"
        embeddings.append(get_embedding(model, tokenizer, text, layer_idx))
    return np.array(embeddings)