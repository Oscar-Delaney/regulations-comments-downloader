import pandas as pd
import os
from PyPDF2 import PdfReader
from openai import OpenAI

# Takes roughly a third of a cent and 10 seconds per pdf to summarize.

# Initialize the OpenAI client
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract text from a single PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() if page.extract_text() else ""
    return text

# Create a good prompt
system_prompt = ("You are a research assistant employed at the Institute for AI Policy and Strategy (IAPS). The Institute for AI Policy and Strategy (IAPS) is a remote-first think tank of aspiring wonks, trying to figure out what risks from advanced AI might matter most, and anticipate them with forward-thinking solutions. We aim to be humble yet purposeful: we’re all having to learn about AI very fast, and we’d love it if you could join us in tackling these risks together. Our work covers three areas: AI policy and standards, compute governance, and international governance & China. Advanced AI systems pose risks that are complex and costly to mitigate, and we need thoughtful, technical work that recognizes both the scale of these potential risks and the uncertainties around them. We work to anticipate these risks and meet them with forward-looking solutions. Your current task is to read through submissions to a US Government proposal from the DEPARTMENT OF COMMERCE National Telecommunications and Information Administration regarding Dual Use Foundation Artificial Intelligence Models With Widely Available Model Weights. IAPS is interested in understanding what other relevant actors have to say about this proposal, so your job is to summarize their submissions.")
# Function to generate a summary of the paper using GPT model
def summarize_paper(content):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize the following text into approximately 50-100 words: {content}"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    
    summary = response.choices[0].message.content.strip()
    return summary

# Main function to iterate through folders, summarize PDFs, and update CSV
def main():
    submissions_folder = r"C:\Users\User\My Drive\Oscar\USG_submissions\regulations-comments-downloader-1\NTIA-2023-0009"
    csv_path = os.path.join(submissions_folder, "comment_details.csv")
    new_csv_path = os.path.join(submissions_folder, "comment_details_with_summaries.csv")
    
    # Read CSV into DataFrame
    df = pd.read_csv(csv_path)
    df['GPT_summary'] = ''  # Add a new column for GPT summaries
    
    # Summarize PDFs and update the DataFrame
    for i, subfolder_name in enumerate(os.listdir(submissions_folder)):
        subfolder_path = os.path.join(submissions_folder, subfolder_name)
        if os.path.isdir(subfolder_path):
            pdf_files = [f for f in os.listdir(subfolder_path) if f.endswith('.pdf')]
            if pdf_files:  # Check if there is at least one PDF
                # Find the row index where the 'id' matches the subfolder name
                row_index = df[df['id'] == subfolder_name].index
                # Proceed only if 'organization' column is not blank for this row
                if not row_index.empty and pd.notna(df.at[row_index[0], 'organization']):
                    pdf_path = os.path.join(subfolder_path, pdf_files[0])
                    try:
                        text = extract_text_from_pdf(pdf_path)[:40000] # truncate to fit context length
                        summary = summarize_paper(text)
                        df.at[row_index[0], 'GPT_summary'] = summary
                    except Exception as e:
                        print(f"Error processing file {pdf_path}: {e}")
                    print(f"{i + 1}/{len(os.listdir(submissions_folder))} done")
    
    # Save the updated DataFrame as a new CSV file
    df.to_csv(new_csv_path, index=False)

if __name__ == "__main__":
    main()
