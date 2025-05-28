import difflib

def get_shorter_text(text1, text2):
    return text1 if len(text1) <= len(text2) else text2

def similarity_score(t1, t2):
    return difflib.SequenceMatcher(None, t1, t2).ratio()

def get_user_choice():
    while True:
        choice = input("Choose better text (1/2/0 for tie): ").strip()
        if choice in {'0', '1', '2'}:
            return int(choice)
        print("Invalid input. Please enter 1, 2, or 0.")

def evaluate_texts(pairs):
    scores = [0, 0]
    for idx, (cand1_a, cand1_b, cand2_a, cand2_b) in enumerate(pairs):
        text1 = get_shorter_text(cand1_a, cand1_b)
        text2 = get_shorter_text(cand2_a, cand2_b)

        sim = similarity_score(text1, text2)
        if sim >= 0.9:
            print(f"Skipping pair {idx + 1} (similarity {sim:.2f})")
            continue

        print(f"\nText 1: {text1}\n\nText 2: {text2}")
        choice = get_user_choice()
        if choice == 1:
            scores[0] += 1
        elif choice == 2:
            scores[1] += 1

    print(f"\nFinal scores:\nText 1: {scores[0]}\nText 2: {scores[1]}")

# # Example usage:
# sample_data = [
#     ("Short A", "Longer text A", "Candidate A", "Candidate AA"),
#     ("Hello world!", "Hello!", "Hi Earth!", "Hi!"),
# ]

# evaluate_texts(sample_data)
