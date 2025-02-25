import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import './index.css';
import { Loader2, Send, Upload, FileText, Trash2, X } from "lucide-react";

function App() {
  // File upload states
  const [files, setFiles] = useState([]);
  const [uploadStatus, setUploadStatus] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Query states
  const [queryInput, setQueryInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);

  // Chat history state
  const [chatHistory, setChatHistory] = useState([]);

  // References
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // API configuration
  const FASTAPI_URL = "http://localhost:8081";

  // Scroll to the bottom of the chat whenever history changes
  useEffect(() => {
    scrollToBottom();
  }, [chatHistory]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Handle file selection
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(prev => [...prev, ...selectedFiles]);
  };

  // Remove a file from the list
  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  // Clear all files
  const clearFiles = () => {
    setFiles([]);
    fileInputRef.current.value = "";
  };

  // Handle file upload
  const handleFileUpload = async (e) => {
    e.preventDefault();
    if (!files.length) return;

    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus("");

    try {
      // Upload files one by one
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const formData = new FormData();
        formData.append("file", file);

        await axios.post(`${FASTAPI_URL}/pdf`, formData, {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
            setUploadProgress(percentCompleted);
          }
        });

        // Update progress for multiple files
        setUploadProgress(Math.round(((i + 1) / files.length) * 100));
      }

      setUploadStatus(`Successfully uploaded ${files.length} file(s)`);

      // Add system message to chat history
      const systemMessage = {
        role: "system",
        text: `${files.length} document(s) uploaded: ${files.map(f => f.name).join(", ")}`
      };

      setChatHistory(prev => [...prev, systemMessage]);
      setFiles([]);
      fileInputRef.current.value = "";

    } catch (err) {
      console.error(err);
      setUploadStatus(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Handle query submission
  const handleQuery = async (e) => {
    e.preventDefault();
    if (!queryInput.trim() || isProcessing) return;

    // Add user message to chat
    setChatHistory(prev => [
      ...prev,
      { role: "user", text: queryInput }
    ]);

    setIsProcessing(true);

    try {
      // Determine which endpoint to use (ask_content for document queries)
      const endpoint = chatHistory.some(msg => msg.role === "system") ?
        "ask_content" : "ai";

      const res = await axios.post(`${FASTAPI_URL}/${endpoint}`, {
        query: queryInput
      });

      // Create assistant message
      const assistantMessage = {
        role: "assistant",
        text: res.data.answer,
        sources: res.data.sources || []
      };

      // Add assistant response to chat
      setChatHistory(prev => [...prev, assistantMessage]);
      setQueryInput("");

    } catch (err) {
      console.error(err);
      // Add error message to chat
      setChatHistory(prev => [
        ...prev,
        {
          role: "error",
          text: `Error: ${err.response?.data?.detail || err.message || "Something went wrong"}`
        }
      ]);
    } finally {
      setIsProcessing(false);
      scrollToBottom();
    }
  };

  // Handle clearing chat history
  const handleClearChat = async () => {
    setChatHistory([]);
    try {
      // Optionally, clear stored embeddings on the server
      await axios.delete(`${FASTAPI_URL}/delete_embeddings`);
    } catch (err) {
      console.error("Failed to clear embeddings:", err);
    }
  };

  // File type icon mapping
  const getFileIcon = (fileName) => {
    const ext = fileName.split('.').pop().toLowerCase();
    switch (ext) {
      case 'pdf':
        return { color: 'text-red-500', icon: <FileText size={16} /> };
      case 'docx':
        return { color: 'text-blue-500', icon: <FileText size={16} /> };
      case 'pptx':
        return { color: 'text-orange-500', icon: <FileText size={16} /> };
      default:
        return { color: 'text-gray-500', icon: <FileText size={16} /> };
    }
  };

  return (
    <div className="min-h-screen flex bg-gray-50">
      {/* Sidebar */}
      <aside className="w-72 bg-gradient-to-b from-blue-900 to-indigo-900 text-white p-5 flex flex-col">
        <h1 className="text-3xl font-bold mb-8 flex items-center">
          <span className="bg-white text-blue-900 p-1 rounded mr-2">S</span>
          amurAI
        </h1>

        <div className="mb-6">
          <h2 className="text-lg font-semibold mb-3">Upload Documents</h2>

          {/* Files display */}
          {files.length > 0 && (
            <div className="mb-4 bg-blue-800 bg-opacity-50 rounded p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm">{files.length} file(s) selected</span>
                <button
                  onClick={clearFiles}
                  className="p-1 hover:bg-blue-700 rounded"
                >
                  <Trash2 size={16} />
                </button>
              </div>

              <div className="space-y-2 max-h-40 overflow-y-auto">
                {files.map((file, index) => {
                  const { color, icon } = getFileIcon(file.name);
                  return (
                    <div key={index} className="flex items-center text-xs bg-blue-800 bg-opacity-30 p-2 rounded">
                      <span className={`mr-2 ${color}`}>{icon}</span>
                      <span className="truncate flex-1">{file.name}</span>
                      <button
                        onClick={() => removeFile(index)}
                        className="ml-2 text-gray-300 hover:text-white"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* File upload form */}
          <form onSubmit={handleFileUpload} className="space-y-3">
            <div className="relative">
              <input
                type="file"
                onChange={handleFileChange}
                className="w-full p-2 rounded bg-blue-800 bg-opacity-50 focus:outline-none focus:ring-2 focus:ring-blue-400 text-sm"
                accept=".pdf,.docx,.pptx"
                multiple
                ref={fileInputRef}
              />
            </div>

            <button
              type="submit"
              disabled={!files.length || isUploading}
              className={`w-full py-2 rounded transition flex items-center justify-center text-sm
                ${files.length && !isUploading ? 'bg-green-500 hover:bg-green-600' : 'bg-gray-500 cursor-not-allowed'}`}
            >
              {isUploading ? (
                <>
                  <Loader2 size={16} className="animate-spin mr-2" />
                  Uploading {uploadProgress}%
                </>
              ) : (
                <>
                  <Upload size={16} className="mr-2" />
                  Upload Document{files.length > 1 ? 's' : ''}
                </>
              )}
            </button>
          </form>

          {uploadStatus && (
            <p className={`mt-2 text-sm ${uploadStatus.includes("failed") ? "text-red-300" : "text-green-300"
              }`}>
              {uploadStatus}
            </p>
          )}
        </div>

        <div className="mt-auto">
          <button
            onClick={handleClearChat}
            className="w-full py-2 bg-blue-800 rounded hover:bg-blue-700 transition text-sm flex items-center justify-center"
          >
            <Trash2 size={16} className="mr-2" />
            Clear Chat & Documents
          </button>

          <div className="mt-6 text-xs text-blue-200 bg-blue-800 bg-opacity-30 p-3 rounded">
            <p>Upload PDFs, DOCX, or PPTX files to ask questions about them.</p>
            <p className="mt-2">© 2025 SamurAI • Powered by FastAPI & LangChain</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white shadow-sm p-4 flex items-center justify-between border-b">
          <h2 className="text-xl font-semibold text-gray-800">
            Interactive Chat
          </h2>
          <div className="text-sm text-gray-500">
            {chatHistory.some(msg => msg.role === "system")
              ? "Document Q&A Mode"
              : "General AI Chat Mode"
            }
          </div>
        </header>

        {/* Chat Area */}
        <main className="flex-1 p-5 overflow-y-auto bg-gray-50">
          {chatHistory.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-gray-400">
              <div className="bg-white p-8 rounded-lg shadow-sm max-w-md text-center">
                <h3 className="text-xl font-medium text-gray-700 mb-2">Welcome to SamurAI</h3>
                <p className="mb-4">Start a conversation or upload documents to ask questions about them.</p>
                <ul className="text-sm text-left space-y-2 mb-4">
                  <li className="flex items-start">
                    <span className="bg-blue-100 text-blue-800 p-1 rounded-full mr-2">1</span>
                    <span>Upload PDFs, Word or PowerPoint documents using the sidebar</span>
                  </li>
                  <li className="flex items-start">
                    <span className="bg-blue-100 text-blue-800 p-1 rounded-full mr-2">2</span>
                    <span>Ask questions in the chat field below</span>
                  </li>
                  <li className="flex items-start">
                    <span className="bg-blue-100 text-blue-800 p-1 rounded-full mr-2">3</span>
                    <span>Get AI-powered answers based on your documents</span>
                  </li>
                </ul>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-6">
              {chatHistory.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === "assistant" || msg.role === "system" || msg.role === "error"
                    ? "justify-start"
                    : "justify-end"
                    }`}
                >
                  <div
                    className={`max-w-xl p-4 rounded-lg shadow-sm
                      ${msg.role === "assistant"
                        ? "bg-white text-gray-800"
                        : msg.role === "user"
                          ? "bg-blue-600 text-white"
                          : msg.role === "system"
                            ? "bg-gray-200 text-gray-800 text-sm"
                            : "bg-red-100 text-red-800 text-sm"
                      }`}
                  >
                    <p className="whitespace-pre-line">{msg.text}</p>

                    {/* Source citations */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                        <p className="font-medium">Sources:</p>
                        <ul className="mt-1 space-y-1">
                          {msg.sources.map((src, i) => (
                            <li key={i} className="bg-gray-50 p-2 rounded">
                              <span className="font-medium">{src.source}:</span> {src.page_content.length > 150
                                ? `${src.page_content.slice(0, 150)}...`
                                : src.page_content}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          )}
        </main>

        {/* Query Input Area */}
        <footer className="bg-white p-4 border-t">
          <form onSubmit={handleQuery} className="max-w-4xl mx-auto">
            <div className="flex items-center">
              <input
                type="text"
                value={queryInput}
                onChange={(e) => setQueryInput(e.target.value)}
                disabled={isProcessing}
                className="flex-1 p-3 border border-gray-300 rounded-l-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder={
                  chatHistory.some(msg => msg.role === "system")
                    ? "Ask a question about your documents..."
                    : "Ask a general question..."
                }
              />
              <button
                type="submit"
                disabled={isProcessing || !queryInput.trim()}
                className={`p-3 rounded-r-lg flex items-center justify-center w-14
                  ${isProcessing || !queryInput.trim()
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700 transition'}`}
              >
                {isProcessing ? (
                  <Loader2 size={20} className="animate-spin" />
                ) : (
                  <Send size={20} />
                )}
              </button>
            </div>
          </form>
        </footer>
      </div>
    </div>
  );
}

export default App;