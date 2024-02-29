
from chatPdfApi import chatPdfApi
from PyPDF2 import PdfReader, PdfWriter
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from rouge import Rouge
from nltk.translate.meteor_score import single_meteor_score


class basicUtils:

    def split_pdf(pdf_path, max_pages_per_file):

        pdfPathList = []

        with open(pdf_path, "rb") as pdf_file:
            pdf = PdfReader(pdf_file)
            total_pages = len(pdf.pages)
            start_page = 0
            file_number = 1

            while start_page < total_pages:
                end_page = min(start_page + max_pages_per_file, total_pages)

                # Create a new PDF writer object for the output PDF
                pdf_writer = PdfWriter()

                # Copy the specified range of pages into the new PDF
                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf.pages[page_num])

                # Save the resulting PDF to a temporary file
                temp_pdf_path = f"temp_split_{file_number}.pdf"
                with open(temp_pdf_path, "wb") as temp_pdf_file:
                    pdf_writer.write(temp_pdf_file)

                # Append the sourceId to the list
                pdfPathList.append(temp_pdf_path)

                # Clean up the temporary file
                # os.remove(temp_pdf_path)

                # Update the start page and file number for the next loop
                start_page = end_page
                file_number += 1

        return pdfPathList


    def calculate_average(numbers):
        total = sum(numbers)

        count = len(numbers)

        return total / count


    def calculate_cosine_similarity(text1, text2):
        corpus = [text1, text2]

        vectorizer = CountVectorizer().fit(corpus)
        vectors = vectorizer.transform(corpus)

        cosine_sim = cosine_similarity(vectors[0], vectors[1])

        return cosine_sim[0][0]

    def tokenize_sentence(sentence):
        return word_tokenize(sentence)

    def get_rouge_scores(hypothesis, reference):
        rouge = Rouge()
        scores = rouge.get_scores(hypothesis, reference, avg=True)
        return scores

    def get_meteor_score(hypothesis, reference):
        # tokenized
        tokenized_reference = basicUtils.tokenize_sentence(reference)
        tokenized_hypothesis = basicUtils.tokenize_sentence(hypothesis)
        return single_meteor_score(tokenized_reference, tokenized_hypothesis)

if __name__ == '__main__':
    basicUtils.split_pdf(10)
