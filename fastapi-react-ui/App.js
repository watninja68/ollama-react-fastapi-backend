import React, { useState } from "react";
import axios from "axios";
import './index.css';

function App() {
  // -- State for file uploads
  const [pdfFile, setPdfFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");

  // -- State for general AI query (/ai)
  const [aiQuery, setAiQuery] = useState("");
  const [aiResponse, setAiResponse] = useState("");

  // -- State for PDF-based query (/ask_content)
  const [pdfQuery, setPdfQuery] = useState("");
  const [pdfResponse, setPdfResponse] = useState("");
  const [pdfSources, setPdfSources] = useState([]);

  const [chatHistory, setChatHistory] = useState([]);

  const FASTAPI_URL = "http://localhost:8081";
  const handlePdfUpload = async (e) => {
    e.preventDefault();
    if (!pdfFile) return;

    const formData = new FormData();
    formData.append("file", pdfFile);

    try {
      const res = await axios.post(${FASTAPI_URL}/pdf, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setUploadStatus(File uploaded: ${res.data.filename});
    } catch (err) {
      console.error(err);
      setUploadStatus("Upload failed.");
    }
  };

  const handleAiQuery = async () => {
    if (!aiQuery) return;
    try {
      const res = await axios.post(${FASTAPI_URL}/ai, { query: aiQuery });
      setAiResponse(res.data.answer);

      // Append to chat history
      setChatHistory((prev) => [
        ...prev,
        { role: "user", text: aiQuery },
        { role: "assistant", text: res.data.answer },
      ]);
      setAiQuery("");
    } catch (err) {
      console.error(err);
      setAiResponse("Error querying AI.");
    }
  };

  // PDF-based query to /ask_content endpoint (updated)
  const handlePdfQuery = async () => {
    if (!pdfQuery) return;
    try {
      const res = await axios.post(${FASTAPI_URL}/ask_content, { query: pdfQuery });
      setPdfResponse(res.data.answer);
      setPdfSources(res.data.sources || []);

      // Append to chat history
      setChatHistory((prev) => [
        ...prev,
        { role: "user", text: pdfQuery },
        { role: "assistant", text: res.data.answer },
      ]);
      setPdfQuery("");
    } catch (err) {
      console.error(err);
      setPdfResponse("Error querying PDF.");
    }
  };

  return (
    <div className="min-h-screen flex bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-gradient-to-b from-blue-800 to-blue-900 text-white p-6 flex flex-col">
        <h1 className="text-3xl font-bold mb-8">SamurAI</h1>
        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-2">Upload Documents</h2>
          <form onSubmit={handlePdfUpload} className="space-y-4">
            <input
              type="file"
              onChange={(e) => setPdfFile(e.target.files[0])}
              className="w-full p-2 rounded bg-blue-700 focus:outline-none"
            />
            <button
              type="submit"
              className="w-full py-2 bg-green-500 rounded hover:bg-green-600 transition"
            >
              Upload
            </button>
          </form>
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
                className={flex ${msg.role === "assistant" ? "justify-start" : "justify-end"}}
              >
                <div
                  className={max-w-md p-4 rounded-lg shadow
                  ${msg.role === "assistant" ? "bg-white text-gray-800" : "bg-blue-500 text-white"}}
                >
                  <p className="text-sm">{msg.text}</p>
                </div>
              </div>
            ))}
          </div>
        </main>

        {/* Query Input Area */}
        <footer className="bg-white p-4 border-t border-gray-200">
          <div className="space-y-6">
            {/* PDF-based Query */}
            <div>
              <label className="block text-sm font-medium text-gray-700">
                Ask a Question about the Uploaded PDF (/ask_content)
              </label>
              <div className="flex mt-1">
                <input
                  type="text"
                  value={pdfQuery}
                  onChange={(e) => setPdfQuery(e.target.value)}
                  className="flex-1 p-2 border border-gray-300 rounded-l focus:outline-none focus:ring-2
focus:ring-blue-500"
                  placeholder="Ask about the PDF..."
                />
                <button
                  onClick={handlePdfQuery}
                  className="px-4 bg-blue-600 text-white rounded-r hover:bg-blue-700 transition"
                >
                  Ask
                </button>
              </div>
              {pdfResponse && (
                <p className="mt-2 text-sm text-green-600">PDF Response: {pdfResponse}</p>
              )}
              {pdfSources.length > 0 && (
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
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
