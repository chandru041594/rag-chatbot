# Example project knowledge

## What is this project?

This project is an Agentic AI RAG chatbot. It searches the project's
knowledge files and uses a language model to answer questions from them.

## How do I run the project?

Install dependencies with npm install. Run it locally with npm run dev.
The local site is served by Netlify Dev.

## What should the chatbot do when it cannot find an answer?

It should say that the available project documents do not contain enough
information. It should not invent facts or claim that an unsupported
statement came from the documents.