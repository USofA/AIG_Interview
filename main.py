import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
import html
import ollama

# Retrieval (for semantic tasks only)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PySpark
from pyspark.sql import SparkSession

# STEP 1: SPARK SESSION
spark = SparkSession.builder \
    .appName("AIG_RAG_Hybrid_Final") \
    .getOrCreate()

# STEP 2: CONFIG
URL = "https://www.sec.gov/Archives/edgar/data/5272/000000527224000023/0000005272-24-000023.txt"
HEADERS = {'User-Agent': 'Francis B. Gallego (francis.b.gallego@gmail.com)'}

VARIABLES = ["Total Revenue", "Total Assets", "CEO"]

GROUND_TRUTH = {
    "Total Revenue": "46802",
    "Total Assets": "539306",
    "CEO": "Peter Zaffino"
}

###########################
# STEP 3: INGESTION + ITEM 8 PRUNING 
#PIPELINE Part 1
###########################
def deep_clean_ingestion(url):
    response = requests.get(url, headers=HEADERS) #retrieve response via get rquest 
    if response.status_code != 200:
        raise Exception("Failed to fetch document")

    text = html.unescape(response.text) #gets rid of and replaces &amp; &quot; &nbsp; &lt; &#160 with &, ", nonbreaking space, less than sign respectively 
    with open("raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    soup = BeautifulSoup(text, "html.parser")  
    for tag in soup(["script", "style"]):
        tag.decompose()  #get rif of all "script" and "style" stuff in html

    full_text = soup.get_text(separator=" ") #strips out all remaining HTML tags <div>, <td>, <tr>, <context>, <period>, <entity>
    full_text = re.sub(r"\s+", " ", full_text).strip() #get any sequence of whitespace (multiple spaces, tabs, newlines) and makes it a SINGLE space 

    # YOUR ORIGINAL HIGH-PRECISION RETRIEVAL
    item_pattern = r'\b(ITEM\s+8\.?)\b' #used to get ITEM 8 #word boundary
    parts = re.split(item_pattern, full_text, flags=re.IGNORECASE) #case insensitive
    item8_text = parts[2] if len(parts) > 2 else full_text #This is where the financial statements actually live - rearranged as 

    return full_text, item8_text

###########################
# STEP 4: CHUNKING (for semantic retrieval only)
###########################
def chunk_text(text, chunk_size=800): #As it stands - only used for CEO name
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)] #chunks based on 800 chars

###########################
# STEP 5: TF-IDF RETRIEVAL (ONLY FOR CEO)
###########################
def retrieve_relevant_chunks(chunks, query, top_k=3):
    vectorizer = TfidfVectorizer() # weight by frequency counts and importance to chunks 
    vectors = vectorizer.fit_transform(chunks + [query]) #essentially make the chunks and query into a matrix vocab

    similarity = cosine_similarity(vectors[-1], vectors[:-1])[0] #closeness betwen query (vectors[-1]) and all chunks vectors[:-1] #gets doc most close
    top_indices = similarity.argsort()[-top_k:][::-1]  #gts indices from highest to lowest similarity and slices the best top_k (3 by default)
    #returns the chunks to be fed into the LLM
    return [chunks[i] for i in top_indices]

###########################
# STEP 6: EXTRACTION (UNCHANGED CORE LOGIC)
###########################
def extract_logic(text_segment, var_type):
    """
    PHASE: REFINED DETERMINISTIC EXTRACTION
    We are expanding the character class to handle decimal precision (periods)
    and thousands separators (commas).
    """

    if var_type == "Total Revenue":
        # Sees Total revenues then any number of spaces then a dollar sign and an optional whitespace 
        # with at least 2 number of characters plus commas and periods
        # Pattern explanation
        # [\d,.]  -> Character class: matches any digit (\d), comma (,), or period (.).
        # {2,}    -> Quantifier: ensures the captured string is at least 2 characters long.
        # s? means s is optional 
        # This will now correctly match "46,802.50", "1.5", or "1,000".

        match = re.search(
            r"total revenues?\s+\$?\s*((?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?)",
            text_segment,
            re.IGNORECASE
        )
        
        # DECISION: We keep the strip logic, but only for commas to preserve the 
        # decimal point for float conversion.
        return match.group(1).replace(',', '') if match else "N/A"

    elif var_type == "Total Assets":
        # Applies the same logic for asset extraction.
        # Sees Total assets then any number of spaces then a dollar sign and an optional whitespace 
        # with at least 2 number of characters plus commas and periods
        match = re.search(r"Total assets\s+\$?\s*([\d,.]{2,})", text_segment, re.IGNORECASE)
        
        return match.group(1).replace(',', '') if match else "N/A"


    
def extract_ceo(text):
    prompt = f"""
    Extract the full name of the Chief Executive Officer (CEO) from this SEC 10-K text.

    Rules:
    - Return ONLY the person's full name
    - No titles, no punctuation, no explanation
    - If multiple roles are listed, choose the one explicitly labeled CEO

    Text:
    {text[:4000]}
    """

    try:
        res = ollama.generate(model="llama3", prompt=prompt)
        raw = res["response"].strip()

        # Clean: keep only letters + spaces
        cleaned = re.sub(r'[^A-Za-z\s]', '', raw).strip()

        # Normalize spacing - anything that's more spaces is just one space
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Basic sanity filter (2–4 words typical for names) William H Smith II as example
        if 2 <= len(cleaned.split()) <= 4:
            return cleaned

        return "N/A"

    except:
        return "N/A"
###########################
# STEP 7: PIPELINE
###########################
def run_pipeline():
    print("Starting Hybrid RAG Pipeline (Best Version)...")

    full_text, item8_text = deep_clean_ingestion(URL)
    # --- NEW: EXPORT FOR EXAMINATION ---
    with open("full_document_scrubbed.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    
    with open("item8_financials_only.txt", "w", encoding="utf-8") as f:
        f.write(item8_text)

    chunks = chunk_text(full_text)
    # CEO-specific chunking
    ceo_chunks = retrieve_relevant_chunks(
        chunks,
        "Chief Executive Officer CEO principal executive officer signature"
    )


    data_rows = []

    print("Running 5 iterations per variable (stability testing)...")

    for var in VARIABLES:

        for i in range(1, 6):

            # STRICT SEPARATION OF LOGIC
            if var == "Total Revenue":
                extracted_val = extract_logic(item8_text, var)

            elif var == "Total Assets":
                extracted_val = extract_logic(item8_text, var)

            elif var == "CEO":
                ceo_text = " ".join(ceo_chunks)
                extracted_val = extract_ceo(ceo_text) 

            lookup_var = var  # original key (DO NOT TOUCH)

            # create display name separately
            if var in ["Total Revenue", "Total Assets"]:
                display_var = var + " (in millions)"
            else:
                display_var = var

            data_rows.append((
                display_var,
                i,
                GROUND_TRUTH[lookup_var],
                extracted_val
            ))

    # --- DATAFRAME ---
    df = spark.createDataFrame(
        data_rows,
        ["Variable", "Observation_ID", "Ground_Truth", "Extracted_Value"]
    )

    # --- EVALUATION ---
    pdf = df.toPandas()
    pdf["Match"] = pdf["Ground_Truth"] == pdf["Extracted_Value"]
    accuracy = pdf["Match"].mean() * 100

    print("\n" + "="*60)
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print("="*60)
    print(pdf)

    pdf.to_csv("AIG_RAG_Results.csv", index=False)
    print("Saved to AIG_RAG_Results.csv")

###########################
# RUN
###########################
if __name__ == "__main__":
    run_pipeline()