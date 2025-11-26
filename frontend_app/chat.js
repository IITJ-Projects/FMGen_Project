class ZevoChat {
  constructor() {
    this.sessionId = this.generateSessionId();
    this.conversationHistory = [];
    this.isRecording = false;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isProcessing = false;

    this.initializeElements();
    this.attachEventListeners();
    this.checkConnection();
    this.autoResizeTextarea();
  }

  generateSessionId() {
    return (
      "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9)
    );
  }

  initializeElements() {
    this.chatMessages = document.getElementById("chatMessages");
    this.messageInput = document.getElementById("messageInput");
    this.sendBtn = document.getElementById("sendBtn");
    this.voiceBtn = document.getElementById("voiceBtn");
    this.typingIndicator = document.getElementById("typingIndicator");
    this.connectionStatus = document.getElementById("connectionStatus");
    this.voiceStatus = document.getElementById("voiceStatus");
    this.charCount = document.getElementById("charCount");
    this.audioPlayer = document.getElementById("audioPlayer");
    this.loadingOverlay = document.getElementById("loadingOverlay");
    this.errorModal = document.getElementById("errorModal");
    this.errorMessage = document.getElementById("errorMessage");
    this.clearChatBtn = document.getElementById("clearChatBtn");
  }

  attachEventListeners() {
    // Send message
    this.sendBtn.addEventListener("click", () => this.sendMessage());
    this.messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Voice recording
    this.voiceBtn.addEventListener("click", () => this.toggleVoiceRecording());

    // Character count
    this.messageInput.addEventListener("input", () => this.updateCharCount());

    // Clear chat
    this.clearChatBtn.addEventListener("click", () => this.clearChat());

    // Error modal
    document
      .getElementById("closeErrorModal")
      .addEventListener("click", () => this.hideError());
    document
      .getElementById("retryBtn")
      .addEventListener("click", () => this.hideError());
  }

  autoResizeTextarea() {
    this.messageInput.addEventListener("input", () => {
      this.messageInput.style.height = "auto";
      this.messageInput.style.height =
        Math.min(this.messageInput.scrollHeight, 120) + "px";
    });
  }

  updateCharCount() {
    const count = this.messageInput.value.length;
    this.charCount.textContent = count;
    this.charCount.className =
      count > 1800 ? "text-xs text-red-500" : "text-xs text-gray-400";
  }

  async checkConnection() {
    try {
      const response = await fetch("/api/health");
      if (response.ok) {
        this.connectionStatus.textContent = "Connected";
        this.connectionStatus.className = "text-sm text-green-500";
      } else {
        throw new Error("Health check failed");
      }
    } catch (error) {
      this.connectionStatus.textContent = "Disconnected";
      this.connectionStatus.className = "text-sm text-red-500";
      console.error("Connection check failed:", error);
    }
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message || this.isProcessing) return;

    this.addMessage(message, "user");
    this.messageInput.value = "";
    this.updateCharCount();
    this.autoResizeTextarea();

    await this.processMessage(message);
  }

  async processMessage(message) {
    this.isProcessing = true;
    this.showTypingIndicator();
    this.disableInput();

    try {
      // Add to conversation history
      this.conversationHistory.push({
        role: "user",
        content: message,
        timestamp: new Date().toISOString(),
      });

      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: message,
          session_id: this.sessionId,
          conversation_history: this.conversationHistory.slice(-10), // Last 10 messages for context
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      // Add AI response to conversation history
      this.conversationHistory.push({
        role: "assistant",
        content: result.response || result.text,
        timestamp: new Date().toISOString(),
      });

      this.hideTypingIndicator();
      this.addMessage(result.response || result.text, "ai");

      // Play audio if available
      if (result.audio_base64) {
        await this.playAudio(result.audio_base64);
      }
    } catch (error) {
      console.error("Error processing message:", error);
      this.hideTypingIndicator();
      this.addMessage("Sorry, I encountered an error. Please try again.", "ai");
      this.showError("Failed to process your message. Please try again.");
    } finally {
      this.isProcessing = false;
      this.enableInput();
    }
  }

  async toggleVoiceRecording() {
    if (this.isRecording) {
      this.stopVoiceRecording();
    } else {
      await this.startVoiceRecording();
    }
  }

  async startVoiceRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.onstop = () => {
        this.processVoiceRecording();
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.updateVoiceUI(true);
    } catch (error) {
      console.error("Error starting voice recording:", error);
      this.showError(
        "Microphone access denied. Please allow microphone access to use voice input."
      );
    }
  }

  stopVoiceRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.mediaRecorder.stream.getTracks().forEach((track) => track.stop());
      this.isRecording = false;
      this.updateVoiceUI(false);
    }
  }

  async processVoiceRecording() {
    const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
    const formData = new FormData();
    formData.append("file", audioBlob, "voice_input.webm");

    this.isProcessing = true;
    this.showTypingIndicator();
    this.disableInput();

    try {
      const response = await fetch("/chat", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();

      // Add to conversation history
      this.conversationHistory.push({
        role: "user",
        content: result.transcribed_text || "Voice message",
        timestamp: new Date().toISOString(),
      });

      this.conversationHistory.push({
        role: "assistant",
        content: result.response || result.text,
        timestamp: new Date().toISOString(),
      });

      this.hideTypingIndicator();

      // Show transcribed text
      if (result.transcribed_text) {
        this.addMessage(result.transcribed_text, "user");
      }

      // Show AI response
      this.addMessage(result.response || result.text, "ai");

      // Play audio if available
      if (result.audio_base64) {
        await this.playAudio(result.audio_base64);
      }
    } catch (error) {
      console.error("Error processing voice recording:", error);
      this.hideTypingIndicator();
      this.addMessage(
        "Sorry, I couldn't process your voice message. Please try again.",
        "ai"
      );
      this.showError("Failed to process voice message. Please try again.");
    } finally {
      this.isProcessing = false;
      this.enableInput();
    }
  }

  updateVoiceUI(recording) {
    if (recording) {
      this.voiceBtn.className =
        "w-10 h-10 bg-green-500 hover:bg-green-600 text-white rounded-full flex items-center justify-center transition-colors flex-shrink-0 animate-pulse";
      this.voiceBtn.innerHTML = '<i class="fas fa-stop text-sm"></i>';
      this.voiceStatus.style.display = "flex";
    } else {
      this.voiceBtn.className =
        "w-10 h-10 bg-red-500 hover:bg-red-600 text-white rounded-full flex items-center justify-center transition-colors flex-shrink-0";
      this.voiceBtn.innerHTML = '<i class="fas fa-microphone text-sm"></i>';
      this.voiceStatus.style.display = "none";
    }
  }

  async playAudio(audioBase64) {
    try {
      const audioData = atob(audioBase64);
      const audioArray = new Uint8Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
      }
      const audioBlob = new Blob([audioArray], { type: "audio/wav" });
      const audioUrl = URL.createObjectURL(audioBlob);

      this.audioPlayer.src = audioUrl;
      await this.audioPlayer.play();

      // Clean up URL after playing
      this.audioPlayer.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
    } catch (error) {
      console.error("Error playing audio:", error);
    }
  }

  addMessage(content, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.className = "flex items-start space-x-3 animate-fade-in";

    if (sender === "user") {
      messageDiv.className += " flex-row-reverse space-x-reverse";
    }

    const avatar = document.createElement("div");
    if (sender === "ai") {
      avatar.className =
        "w-8 h-8 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full flex items-center justify-center flex-shrink-0";
      avatar.innerHTML = '<i class="fas fa-robot text-white text-sm"></i>';
    } else {
      avatar.className =
        "w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0";
      avatar.innerHTML = '<i class="fas fa-user text-gray-600 text-sm"></i>';
    }

    const messageContent = document.createElement("div");
    messageContent.className = "flex-1";
    if (sender === "user") {
      messageContent.className += " flex flex-col items-end";
    }

    const messageBubble = document.createElement("div");
    if (sender === "ai") {
      messageBubble.className =
        "bg-gray-100 rounded-2xl rounded-tl-md px-4 py-3 max-w-3xl";
    } else {
      messageBubble.className =
        "bg-primary-500 text-white rounded-2xl rounded-tr-md px-4 py-3 max-w-3xl";
    }

    messageBubble.innerHTML = `<p class="leading-relaxed">${this.escapeHtml(
      content
    )}</p>`;

    const messageTime = document.createElement("p");
    messageTime.className = "text-xs text-gray-500 mt-1";
    if (sender === "user") {
      messageTime.className += " text-right";
    } else {
      messageTime.className += " ml-4";
    }
    messageTime.textContent = this.formatTime(new Date());

    messageContent.appendChild(messageBubble);
    messageContent.appendChild(messageTime);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    this.chatMessages.appendChild(messageDiv);
    this.scrollToBottom();
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  showTypingIndicator() {
    this.typingIndicator.style.display = "block";
    this.scrollToBottom();
  }

  hideTypingIndicator() {
    this.typingIndicator.style.display = "none";
  }

  scrollToBottom() {
    setTimeout(() => {
      this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }, 100);
  }

  formatTime(date) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  disableInput() {
    this.messageInput.disabled = true;
    this.sendBtn.disabled = true;
    this.voiceBtn.disabled = true;
  }

  enableInput() {
    this.messageInput.disabled = false;
    this.sendBtn.disabled = false;
    this.voiceBtn.disabled = false;
    this.messageInput.focus();
  }

  clearChat() {
    if (confirm("Are you sure you want to clear the chat history?")) {
      this.chatMessages.innerHTML = `
        <div class="flex items-start space-x-3">
          <div class="w-8 h-8 bg-gradient-to-r from-primary-500 to-primary-600 rounded-full flex items-center justify-center flex-shrink-0">
            <i class="fas fa-robot text-white text-sm"></i>
          </div>
          <div class="flex-1">
            <div class="bg-gray-100 rounded-2xl rounded-tl-md px-4 py-3 max-w-3xl">
              <p class="text-gray-800 leading-relaxed">
                Hello! I'm Zevo AI, your conversational assistant. I can help you with questions, have conversations, and assist with various tasks. You can type your message or use voice input. How can I help you today?
              </p>
            </div>
            <p class="text-xs text-gray-500 mt-1 ml-4">Just now</p>
          </div>
        </div>
      `;
      this.conversationHistory = [];
      this.sessionId = this.generateSessionId();
    }
  }

  showError(message) {
    this.errorMessage.textContent = message;
    this.errorModal.style.display = "flex";
  }

  hideError() {
    this.errorModal.style.display = "none";
  }
}

// Add custom CSS for animations
const style = document.createElement("style");
style.textContent = `
  @keyframes fade-in {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .animate-fade-in {
    animation: fade-in 0.3s ease-out;
  }
`;
document.head.appendChild(style);

// Initialize chat when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  new ZevoChat();
});
