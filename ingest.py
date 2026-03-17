import os
import pandas as pd
import chromadb
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# ── CONFIG ───────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY     = os.getenv("OPENAI_API_KEY")
XLSX_PATH = Path("FamilyOffice_Intelligence_Dataset.xlsx")
DB_PATH     = str(Path(__file__).parent / "chroma_db")
EMBED_MODEL = "text-embedding-3-small"

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Make sure .env file exists with your key.")

client = OpenAI(api_key=API_KEY)

# ── LOAD DATASET ─────────────────────────────────────────────────────────────
print("Loading dataset...")
if not XLSX_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {XLSX_PATH}\nMake sure FamilyOffice_Intelligence_Dataset.xlsx is in the Task 1 folder.")

df = pd.read_excel(XLSX_PATH, sheet_name="Family Office Intelligence")
df = df.fillna("")
print(f"Loaded {len(df)} records with {len(df.columns)} columns")

# ── CONVERT EACH ROW TO A RICH TEXT CHUNK ────────────────────────────────────
def row_to_text(row):
    return f"""
Family Office: {row['FO Firm Name']}
Type: {row['FO Type (SFO/MFO)']}
AUM: {row['AUM Range']}
Location: {row['City/Location']}, {row['Country']}
Website: {row['FO Website URL']}
Founding Family / Source of Wealth: {row['Founding Family / Source of Wealth']}
Investment Thesis: {row['Investment Thesis / Description']}
Investment Strategy: {row['Investment Strategy Summary']}
Sector Focus: {row['Sector Focus']}
Investment Vehicles: {row['Investment Vehicles']}
Check Size Range: {row['Acceptable Check Size Range']}
Investment Style: {row['Investment Style']}
Co-Invest Frequency: {row['Co-Invest Frequency']}
Notable Portfolio Companies: {row['Notable Portfolio Companies']}
Key Decision Maker: {row['Key Decision Maker - First Name']} {row['Key Decision Maker - Last Name / Role']}
Contact Title: {row['Contact Title']}
Contact Email: {row['Contact Email (Primary)']}
Contact LinkedIn: {row['Contact LinkedIn']}
Direct Investment Active: {row['Direct Investment Active (Y/N)']}
Last Active Investment Year: {row['Last Active Investment (Year)']}
Recent News Signal: {row['Recent Filings / News Signal']}
Recent LinkedIn Activity: {row['Recent LinkedIn Activity']}
Data Source: {row['Data Source(s)']}
Validation Status: {row['Validation Status']}
Notes: {row['Notes / Additional Intelligence']}
""".strip()

# ── EMBED IN BATCHES ─────────────────────────────────────────────────────────
def get_embeddings(texts, batch_size=50):
    all_embeddings = []
    total_batches = (len(texts) - 1) // batch_size + 1
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Embedding batch {i // batch_size + 1}/{total_batches} ({len(batch)} records)...")
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=batch
        )
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings

# ── BUILD CHROMADB COLLECTION ────────────────────────────────────────────────
print("\nSetting up ChromaDB...")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Check if already ingested
existing = chroma_client.list_collections()
existing_names = [c.name for c in existing]

if "family_offices" in existing_names:
    collection = chroma_client.get_collection("family_offices")
    if collection.count() == len(df):
        print(f"Collection already exists with {collection.count()} records. Skipping ingestion.")
        print("Delete the chroma_db folder manually if you want to re-ingest.")
        exit(0)
    else:
        print(f"Collection exists but has {collection.count()} records (expected {len(df)}). Rebuilding...")
        chroma_client.delete_collection("family_offices")
        collection = chroma_client.create_collection(
            name="family_offices",
            metadata={"hnsw:space": "cosine"}
        )
else:
    collection = chroma_client.create_collection(
        name="family_offices",
        metadata={"hnsw:space": "cosine"}
    )

# ── PREPARE DATA ─────────────────────────────────────────────────────────────
print("\nPreparing text chunks...")
texts     = [row_to_text(row) for _, row in df.iterrows()]
ids       = [f"fo_{i}" for i in range(len(df))]
metadatas = [
    {
        "name":     str(row["FO Firm Name"]),
        "type":     str(row["FO Type (SFO/MFO)"]),
        "aum":      str(row["AUM Range"]),
        "country":  str(row["Country"]),
        "sector":   str(row["Sector Focus"]),
        "email":    str(row["Contact Email (Primary)"]),
        "check":    str(row["Acceptable Check Size Range"]),
        "coinvest": str(row["Co-Invest Frequency"]),
        "direct":   str(row["Direct Investment Active (Y/N)"]),
        "style":    str(row["Investment Style"]),
    }
    for _, row in df.iterrows()
]

# ── EMBED AND STORE ───────────────────────────────────────────────────────────
print("\nGenerating embeddings (this takes ~1-2 minutes)...")
embeddings = get_embeddings(texts)

print("\nStoring in ChromaDB...")
collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print(f"\nDone! {collection.count()} records stored in ChromaDB at '{DB_PATH}'")
print("Next step: py -m streamlit run app.py")