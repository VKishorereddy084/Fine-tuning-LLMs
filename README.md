# Question generation app
Install the all necessary libraries. You need to download the zephyr model (TheBloke/zephyr-7B-alpha-GGUF) from Hugging Face.
When you run the code, you will be asked to give path of the pdf file. Please give the correct path.
Its better to give 5-6 page pdf as input at a time. Because it will take lot of time to generate QA pairs depends upon the system RAM.(Select 5-6 pages from the whole pdf and save it as another pdf and pass this as input).  If you give more pages as input, it may lead to crashing the code.

Link to the book: (https://oercommons.org/courses/python-for-everybody-exploring-data-in-python-3). 

Reference: (https://github.com/InsightEdge01/Question-AnswerPairGeneratorApp/blob/master/app.py)

# Push to Hub
Generated QA needs to be  convert to the format suitable for llama2 prompt template.
It takes csv file as input.

# Fine-tuning
To use Llama2 model (meta-llama/Llama-2-7b-chat-hf) you need to request access from Meta AI
To use my dataset or model, you need to login to your HF and serach Kishorereddy123, you will find the datasets and models.
After fine-tuning, to store the new model we need to merge the weights from LoRA with the base model. Unfortunately, there is no straightforward way to do it: we need to reload the base model in FP16 precision and use the peft library to merge everything. Clear the VRAM before merging and run the first 3 cells again.
For evaluation, test dataset have been used. You can use my accurate_QA from HF.

Reference : (https://github.com/amitsangani/Llama/blob/main/Llama_2_Fine_Tune_With_Your_Dataset.ipynb)

# RAG
To run this code, you need to install (llama-2-7b-chat.Q4_K_M.gguf) from hugging face.

Reference: (https://github.com/AIAnytime/Llama2-Medical-Chatbot)
