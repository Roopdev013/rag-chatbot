code = open('app.py', 'r').read()
code = code.replace(
    'if all_chunks:\n                Chroma.from_documents(documents=all_chunks,embedding=embeddings,persist_directory="chroma_db")\nelse:\n                st.error("Could not extract text from PDF. Try a different file.")',
    'if all_chunks:\n                    Chroma.from_documents(documents=all_chunks,embedding=embeddings,persist_directory="chroma_db")\n                else:\n                    st.error("Could not extract text from PDF. Try a different file.")'
)
open('app.py', 'w').write(code)
print("Fixed!")