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
        "Benjamin", "Bruce", "Brandon", "Donald", "Scott", "Dennis", "Patrick", "Alexander", "Raymond", "Gregory",
        "Andrew", "Anthony", "Austin", "Billy", "Bradley", "Bryan", "Carl", "Charles", "Christian", "Christopher",
        "Daniel", "David", "Dylan", "Eric", "Ethan", "Eugene", "Gabriel", "Gary", "Howard", "Isaac",
        "James", "Jason", "Jesse", "Joe", "Jordan", "Joseph", "Joshua", "Juan", "Kenneth", "Kyle",
        "Liam", "Logan", "Louis", "Mark", "Matthew", "Michael", "Nathan", "Noah", "Oscar", "Philip",
        "Randy", "Richard", "Robert", "Ronald", "Sean", "Stephen", "Timothy", "Victor", "Wayne", "William"
    ]
FEMALE_NAMES = [
        "Mary", "Jennifer", "Lisa", "Michelle", "Sarah", "Kim", "Amy", "Susan", "Rebecca", "Anna",
        "Emily", "Elizabeth", "Alice", "Amanda", "Laura", "Cynthia", "Linda", "Jessica", "Patricia", "Barbara",
        "Angela", "Christine", "Debra", "Rachel", "Janet", "Catherine", "Maria", "Heather", "Diane", "Virginia",
        "Julie", "Joyce", "Victoria", "Kelly", "Lauren", "Martha", "Judith", "Cheryl", "Megan", "Andrea",
        "Ann", "Evelyn", "Jean", "Kathryn", "Jacqueline", "Hannah", "Carol", "Gloria", "Teresa", "Sara",
        "Abigail", "Alexandra", "Allison", "Amber", "Ashley", "Beverly", "Brenda", "Brittany", "Carolyn", "Cassandra",
        "Chloe", "Christina", "Crystal", "Danielle", "Denise", "Donna", "Doris", "Dorothy", "Frances", "Grace",
        "Isabella", "Jane", "Janice", "Joan", "Josephine", "Judy", "Julia", "Karen", "Katherine", "Kathleen",
        "Kayla", "Kimberly", "Madison", "Margaret", "Marie", "Melissa", "Mia", "Nancy", "Natalie", "Nicole",
        "Olivia", "Pamela", "Rose", "Samantha", "Sharon", "Stephanie", "Taylor", "Vanessa", "Wendy", "Zoe"
    ]

EA_NAMES = [
        "Greg", "Emily", "Anne", "Jill", "Todd", "Neil", "Geoffrey", "Brett", "Brendan", "Laurie",
        "Brad", "Claire", "Colin", "Carrie", "Dustin", "Ellen", "Heather", "Hunter", "Holly", "Jake",
        "Jenna", "Josh", "Katy", "Kristen", "Molly", "Peter", "Sarah", "Scott", "Stephanie", "Cody"
    ]
AA_NAMES = [
        "Lakisha", "Jamal", "Darnell", "Tyrone", "Latoya", "Ebony", "Keisha", "Jermaine", "Tanisha", "DeShawn",
        "Aaliyah", "Andre", "Darius", "Diamond", "Hakim", "Imani", "Jada", "Jalen", "Kareem", "Kendrick",
        "Malik", "Marquis", "Nia", "Precious", "Quinton", "Rashad", "Terrence", "Tremayne", "Xavier", "Yolanda"
    ]
def load_model():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, trust_remote_code=False
    ).to(DEVICE)
    model.eval()
    return model, tokenizer

def get_layered_embeddings(model, tokenizer, text, layers=[0, 16, -1]):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    results = {}
    for idx in layers:
        # Map -1 to the actual last layer index
        actual_idx = idx if idx != -1 else len(outputs.hidden_states) - 1
        results[idx] = outputs.hidden_states[actual_idx][0, -1, :].cpu().float().numpy()
    return results

def get_dataset_embeddings(model, tokenizer, names, prompt_template, layer_idx=-1):
    embeddings = []
    for name in names:
        text = f"<|system|>\n{prompt_template}<|end|>\n<|user|>\n{name}<|end|>\n<|assistant|>"
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        embeddings.append(outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy())
    return np.array(embeddings)