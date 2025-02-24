import React, { useState, useRef } from "react";
import axios from "axios";
import './index.css';

function App() {
  // -- State for file uploads
  const [pdfFile, setPdfFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  // -- Combined state for the query input and response
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [pdfSources, setPdfSources] = useState([]);

  const [chatHistory, setChatHistory] = useState([]);

  // -- State to choose query mode: "without" for general query, "with" for PDF-based query (RAG)
  const [queryMode, setQueryMode] = useState("without");

  const FASTAPI_URL = "http://localhost:8081";

  // Reference to the hidden file input
  const fileInputRef = useRef(null);

  // Handle file selection via input
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setPdfFile(e.target.files[0]);
    }
  };

  // Handle drag over event
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // Handle drop event
  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setPdfFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  };

  // Open file picker when clicking the drag-and-drop area
  const handleAreaClick = () => {
    fileInputRef.current.click();
  };

  const handlePdfUpload = async (e) => {
    e.preventDefault();
    if (!pdfFile) return;

    const formData = new FormData();
    formData.append("file", pdfFile);

    try {
      const res = await axios.post(`${FASTAPI_URL}/pdf`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadStatus(`File uploaded: ${res.data.filename}`);
    } catch (err) {
      console.error(err);
      setUploadStatus("Upload failed.");
    }
  };

  const handleQuery = async () => {
    if (!query) return;
    try {
      let res;
      if (queryMode === "without") {
        // General AI query without RAG
        res = await axios.post(`${FASTAPI_URL}/ai`, { query });
      } else {
        // PDF-based query with RAG (/ask_content)
        res = await axios.post(`${FASTAPI_URL}/ask_content`, { query });
        setPdfSources(res.data.sources || []);
      }
      setResponse(res.data.answer);

      // Append the conversation to chat history
      setChatHistory((prev) => [
        ...prev,
        { role: "user", text: query },
        { role: "assistant", text: res.data.answer },
      ]);
      setQuery("");
    } catch (err) {
      console.error(err);
      setResponse("Error querying AI.");
    }
  };

  return (
    <div className="min-h-screen flex bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-gradient-to-b from-blue-800 to-blue-900 text-white p-6 flex flex-col">
        <h1 className="text-3xl font-bold mb-8">SamurAI</h1>
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-2">Upload Documents</h2>
          <div
            className="w-full p-4 border-2 border-dashed border-gray-300 rounded bg-blue-700 hover:bg-blue-600 cursor-pointer text-center"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={handleAreaClick}
          >
            {pdfFile ? (
              <p className="text-white">
                Selected File: {pdfFile.name}
              </p>
            ) : (
              <p className="text-white">
                Drag and drop your PDF here, or click to select a file.
              </p>
            )}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
            />
          </div>
          <button
            onClick={handlePdfUpload}
            className="mt-4 w-full py-2 bg-green-500 rounded hover:bg-green-600 transition"
          >
            Upload
          </button>
          {uploadStatus && (
            <p className="mt-2 text-sm text-green-300">{uploadStatus}</p>
          )}
        </div>
        <div className="mt-auto text-xs text-blue-200">
          <p>Drag and drop files, or browse to upload.</p>
          <p className="mt-2">Powered by FastAPI & LangChain</p>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white shadow p-4 flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-800">
            Ask Questions about Your Documents
          </h2>
        </header>

        {/* Chat Area */}
        <main className="flex-1 p-6 overflow-y-auto">
          <div className="space-y-4">
            {chatHistory.map((msg, idx) => (
              <div
                key={idx}
                className={`flex ${msg.role === "assistant" ? "justify-start" : "justify-end"}`}
              >
                <div
                  className={`max-w-md p-4 rounded-lg shadow ${
                    msg.role === "assistant" ? "bg-white text-gray-800" : "bg-blue-500 text-white"
                  }`}
                >
                  <p className="text-sm">{msg.text}</p>
                </div>
              </div>
            ))}
          </div>
          {response && (
            <div className="mt-4 p-4 bg-green-100 rounded">
              <p className="text-sm">{response}</p>
              {queryMode === "with" && pdfSources.length > 0 && (
                <div className="mt-2 text-xs text-gray-500">
                  <p>Sources:</p>
                  <ul className="list-disc pl-5">
                    {pdfSources.map((src, i) => (
                      <li key={i}>
                        <strong>{src.source}:</strong> {src.page_content.slice(0, 100)}...
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </main>

        {/* Query Input Area */}
        <footer className="bg-white p-4 border-t border-gray-200">
          <div className="flex items-center space-x-2">
            <select
              value={queryMode}
              onChange={(e) => setQueryMode(e.target.value)}
              className="p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="without">General Query (without RAG)</option>
              <option value="with">PDF Query (with RAG)</option>
            </select>
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Enter your question..."
            />
            <button
              onClick={handleQuery}
              className="px-4 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              Ask
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
