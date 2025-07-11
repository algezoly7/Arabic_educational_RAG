<p align="center">
  <img width="375" height="250" alt="ChatGPT Image Jul 11, 2025, 03_23_27 AM" src="https://github.com/user-attachments/assets/a6c65a7b-3afc-4c25-93f5-81cd2651b76a" />
</p>

<h1>Introducing Arabic Educational Rag</h1>
<h4>A RAG for the next generation</h4>
<p>Retrieval-augmented generation (RAG) is a technique that enables large language models (LLMs) to retrieve and incorporate new information. Using this strategy, LLMs do not respond to user queries until they refer to a specified set of documents.
  Instead, they first retrieve relevant documents from a specified corpus and then generate responses grounded in that retrieved content.
  Rather than relying on memorized information, the model extracts pertinent details from the source texts and presents them to the user in a coherent and readable format.</p>

<p>Using this technique, I developed the Arabic Educational RAG, a system designed to solve the problem of quick and easy access to school textbooks.
  With this tool, students no longer need to manually search through their books for every piece of information they need, nor do they have to rely solely on teachers for answers.
  Instead, they can simply ask the RAG system, which retrieves relevant content from the textbooks and presents it clearly and concisely. 
  Also, as a bonus, the teacher can access all students' questions and AI answers, and he can modify the answer he thinks is inaccurate.
  Consequently, the script checks the similarity between the query and questions modified by the teacher, and it decides to give the teacher or the AI answer based on the similarity scores</p>

<h1>Implementation details</h1>
<p>Regarding the implementation of the project, every rag system needs a vector store, as we don't need the whole document for every question, but we only need specific parts of the document relevant to our question.
  So we divide our document into small chunks and transform every chunk into a vector of numbers (embeddings). Searching through these chunks can be relatively easy using mathematical techniques such as cosine similarity or Euclidean distance. 
  In the implementation, I use FAISS (Facebook AI Similarity Search) vector store imported from langchain. <b>The script automatically detects whether a document has already been transformed into a vector store and takes the appropriate action accordingly.</b></p>
<p>As the brain of the system, I used <b>gemma 3n 4b</b> mode, which is a lightweight and cheap LLM suitable for our application. I also incorporated a token calculation system, as this platform is intended to serve schools and businesses. Each school and student will be assigned a quota or usage limit.</p>
<p>Additionally, to ensure easy integration with any project, the system uses clear JSON input and output formats, as demonstrated in the attached.</p>
<h2>Examples from the system</h2>

```
User query: بصورة عامة، عن ماذا يتحدث الكتاب المرفق؟
AI anser: يتحدث الكتاب بشكل عام عن مفاهيم الكمبيوتر، مثل أزرار الفأرة والطابعة والبرامج التطبيقية المستخدمة في مختلف مجالات الحياة
```
```
User query: ما هي الفأرة (mouse)?
AI anser: الفأرة هي جهاز تأشير هام يتحكم بموقع الإشارة (السهم) على جهاز الإخراج (الشاشة). وتتم عملية إدخال الأوامر عن طريق النقر أو السحب. واهم وظائف الفأرة هي (التحريك، التأشير، السحب، النقر بالزر الايسر، النقر بالزر الايمن).
```
