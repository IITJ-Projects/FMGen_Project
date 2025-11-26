class ZevoChat {
  constructor() {
    this.sessionId = this.generateSessionId();
    this.conversationHistory = [];
    this.isRecording = false;
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.isProcessing = false;
    this.websocket = null;
    this.isStreaming = false;
    this.currentStreamingMessage = null;

    // Initialize audio-related properties
    this.audioContext = null;
    this.audioQueue = [];
    this.scheduledSources = [];
    this.nextPlayTime = 0;
    this.isScheduling = false;
    this.isPlayingAudio = false;

    // Audio optimization properties
    this.audioCodec = "wav"; // Default, will be updated from server
    this.sampleRate = 16000; // Optimized for speech
    this.chunkDuration = 30; // 30ms chunks
    this.adaptiveBufferSize = 3; // Buffer 3 chunks ahead
    this.networkLatency = 0; // Track network latency
    this.audioQuality = "high"; // high, medium, low

    // WebRTC properties
    this.webrtcConnection = null;
    this.dataChannel = null;
    this.useWebRTC = false; // Toggle for WebRTC vs WebSocket
    this.iceServers = [
      { urls: "stun:stun.l.google.com:19302" },
      { urls: "stun:stun1.l.google.com:19302" },
    ];

    // Voice Activity Detection properties
    this.vadEnabled = true;
    this.isUserSpeaking = false;
    this.silenceThreshold = 0.01; // Volume threshold for silence detection
    this.speechTimeout = 2000; // 2 seconds of silence to stop recording
    this.analyser = null;
    this.microphone = null;
    this.audioDataArray = null;

    // Mode management
    this.currentMode = "text"; // "text" or "voice"
    this.isVoiceModeActive = false;
    this.voiceModeRecording = false;

    // Call management
    this.isInCall = false;
    this.callStartTime = null;
    this.callDurationInterval = null;
    this.isMuted = false;
    this.isOnHold = false;
    this.continuousListening = false;
    this.restartListeningTimeout = null;
    this.audioPlaying = false;

    // Text Mode behavior flags
    this.useStreamingText = true; // Enable streaming in Text Mode (WebSocket tokens + optional TTS)
    this.readAloudEnabled = false; // User-controlled read aloud
    this.showLatencyReport = false; // Hide performance report in chat responses

    this.initializeElements();
    this.attachEventListeners();
    this.autoResizeTextarea();

    // Initialize connections in parallel for faster startup
    this.initializeConnections();

    // Immediate health check for faster status display
    this.checkConnection();

    // Set up periodic health checks (reduced frequency)
    this.healthCheckInterval = setInterval(() => {
      this.checkConnection();
    }, 300000); // Check every 5 minutes to reduce log spam
  }

  generateSessionId() {
    return (
      "session_" + Date.now() + "_" + Math.random().toString(36).substr(2, 9)
    );
  }

  async initializeConnections() {
    console.log("🚀 Starting parallel connection initialization...");
    const startTime = Date.now();

    // Set a global connection timeout to prevent UI from staying in "Connecting" state
    this.connectionTimeout = setTimeout(() => {
      if (
        this.connectionStatus &&
        this.connectionStatus.textContent.includes("Connecting")
      ) {
        console.warn("⚠️ Connection timeout, forcing status update");
        this.updateConnectionStatus("connected"); // Assume connected if timeout
      }
    }, 8000); // 8-second timeout

    // Initialize all connections in parallel for faster startup
    const connectionPromises = [
      this.initializeWebSocket(),
      this.initializeWebRTC(),
      this.initializeVAD(),
    ];

    try {
      const results = await Promise.allSettled(connectionPromises);
      const endTime = Date.now();
      const duration = endTime - startTime;

      console.log(`✅ Connection initialization completed in ${duration}ms`);

      // Clear connection timeout since we're done
      if (this.connectionTimeout) {
        clearTimeout(this.connectionTimeout);
        this.connectionTimeout = null;
      }

      // Handle results
      results.forEach((result, index) => {
        const names = ["WebSocket", "WebRTC", "VAD"];
        if (result.status === "fulfilled") {
          console.log(`✅ ${names[index]} initialized successfully`);
        } else {
          console.warn(
            `⚠️ ${names[index]} initialization failed:`,
            result.reason
          );
        }
      });

      // Create WebRTC offer if WebRTC was successful
      if (results[1].status === "fulfilled" && results[1].value) {
        this.createWebRTCOffer();
      }
    } catch (error) {
      console.error("❌ Connection initialization failed:", error);
    }
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

    // Optional read-aloud toggle if present in DOM
    this.readAloudToggle = document.getElementById("readAloudToggle");
    if (this.readAloudToggle) {
      this.readAloudEnabled = !!this.readAloudToggle.checked;
      this.readAloudToggle.addEventListener("change", () => {
        this.readAloudEnabled = !!this.readAloudToggle.checked;
      });
    }

    // Mode elements
    this.modeToggle = document.getElementById("modeToggle");
    this.modeToggleThumb = document.getElementById("modeToggleThumb");
    this.textInputArea = document.getElementById("textInputArea");
    this.voiceInputArea = document.getElementById("voiceInputArea");
    this.voiceModeBtn = document.getElementById("voiceModeBtn");
    this.voiceModeStatus = document.getElementById("voiceModeStatus");
    this.voiceModeSubStatus = document.getElementById("voiceModeSubStatus");

    // Call interface elements
    this.startCallInterface = document.getElementById("startCallInterface");
    this.activeCallInterface = document.getElementById("activeCallInterface");
    this.callStatusIndicator = document.getElementById("callStatusIndicator");
    this.callStatusText = document.getElementById("callStatusText");
    this.callDuration = document.getElementById("callDuration");
    this.callMuteBtn = document.getElementById("callMuteBtn");
    this.callHoldBtn = document.getElementById("callHoldBtn");
    this.endCallBtn = document.getElementById("endCallBtn");
    this.callInfo = document.getElementById("callInfo");
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

    // Mode toggle
    this.modeToggle.addEventListener("click", () => this.toggleMode());

    // Voice mode button
    this.voiceModeBtn.addEventListener("click", () => this.toggleVoiceMode());

    // Voice mode controls removed - using call interface instead

    // Call controls (with null checks)
    if (this.callMuteBtn) {
      this.callMuteBtn.addEventListener("click", () => this.toggleCallMute());
    }
    if (this.callHoldBtn) {
      this.callHoldBtn.addEventListener("click", () => this.toggleCallHold());
    }
    if (this.endCallBtn) {
      this.endCallBtn.addEventListener("click", () => this.endCall());
    }

    // Error modal
    document
      .getElementById("closeErrorModal")
      .addEventListener("click", () => this.hideError());
    document
      .getElementById("retryBtn")
      .addEventListener("click", () => this.hideError());

    // Keyboard shortcuts
    document.addEventListener("keydown", (e) =>
      this.handleKeyboardShortcuts(e)
    );
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
      const response = await fetch("/health");
      if (response.ok) {
        this.updateConnectionStatus("connected");
        console.log("Health check successful - connected to backend");
      } else {
        throw new Error("Health check failed");
      }
    } catch (error) {
      this.updateConnectionStatus("disconnected");
      console.error("Connection check failed:", error);
    }
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    console.log(
      "sendMessage called with:",
      message,
      "isProcessing:",
      this.isProcessing,
      "isStreaming:",
      this.isStreaming
    );

    if (!message || this.isProcessing || this.isStreaming) {
      console.log("Message empty or processing, returning");
      return;
    }

    console.log("Sending message:", message);
    this.messageInput.value = "";
    this.updateCharCount();
    this.autoResizeTextarea();

    // Enforce streaming-only text mode: require WebSocket, no REST fallback
    if (!this.useStreamingText) {
      console.warn(
        "Streaming disabled flag is off; refusing to send in Text Mode"
      );
      this.showError(
        "Streaming is disabled. Please enable streaming to send messages."
      );
      return;
    }

    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket not connected - cannot stream message");
      this.showError("Connection lost. Please refresh the page to re-connect.");
      return;
    }

    console.log("Using WebSocket streaming for text message");
    this.sendStreamingMessage(message);
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

      console.log("Sending request to /api/chat with:", {
        message: message,
        session_id: this.sessionId,
        conversation_history: this.conversationHistory.slice(-10),
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
          read_aloud: this.readAloudEnabled,
        }),
      });

      console.log("Response status:", response.status);
      console.log("Response headers:", response.headers);

      if (!response.ok) {
        const errorText = await response.text();
        console.error("Error response:", errorText);
        throw new Error(
          `HTTP ${response.status}: ${response.statusText} - ${errorText}`
        );
      }

      const result = await response.json();
      console.log("Response result:", result);

      // Log latency report if available
      if (this.showLatencyReport && result.latency_report) {
        console.log("📊 Latency Report:", result.latency_report);
        this.displayLatencyReport(result.latency_report);
      }

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
      console.log("🎤 Starting voice recording...");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      console.log("🎤 Microphone stream obtained");

      // Configure MediaRecorder for better audio quality
      const options = {
        mimeType: "audio/webm;codecs=opus",
        audioBitsPerSecond: 128000, // Higher bitrate for better quality
      };

      // Fallback to default if opus is not supported
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = "audio/webm";
        console.log("⚠️ Opus codec not supported, using default WebM");
      }

      this.mediaRecorder = new MediaRecorder(stream, options);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        console.log("🎤 Audio data available, chunk size:", event.data.size);
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        } else {
          console.warn("⚠️ Empty audio chunk received");
        }
      };

      this.mediaRecorder.onstop = () => {
        console.log("🎤 Recording stopped, processing voice...");
        this.processVoiceRecording();
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.updateVoiceUI(true);
      console.log("🎤 Voice recording started successfully");
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

  // Alias functions for call mode compatibility
  async startRecording() {
    console.log(
      "🎤 startRecording() called - delegating to startVoiceRecording()"
    );
    return await this.startVoiceRecording();
  }

  stopRecording() {
    return this.stopVoiceRecording();
  }

  async processVoiceRecording() {
    // Don't process if call has ended
    if (!this.isInCall && this.currentMode === "voice") {
      console.log("📞 Call ended - ignoring voice processing");
      return;
    }

    const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
    console.log("🎤 Audio blob created, size:", audioBlob.size, "bytes");
    console.log("🎤 Audio chunks count:", this.audioChunks.length);

    // Check if we have valid audio data
    if (audioBlob.size === 0) {
      console.error("❌ No audio data captured");
      this.hideTypingIndicator();
      this.enableInput();
      return;
    }

    this.isProcessing = true;
    this.showTypingIndicator();
    this.disableInput();

    console.log(
      "🎤 Processing voice recording - Current mode:",
      this.currentMode
    );
    console.log("🎤 WebRTC connection:", !!this.webrtcConnection);
    console.log(
      "🎤 Data channel:",
      !!this.dataChannel,
      this.dataChannel?.readyState
    );
    console.log("🎤 WebSocket:", !!this.websocket, this.websocket?.readyState);

    try {
      // Use WebRTC for voice calls (phone-like conversation)
      if (
        this.currentMode === "voice" &&
        this.webrtcConnection &&
        this.dataChannel &&
        this.dataChannel.readyState === "open"
      ) {
        console.log("🎤 Using WebRTC for voice call processing");
        await this.processVoiceWithWebRTC(audioBlob);
      }
      // Use WebSocket for text mode voice input
      else if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        console.log("🎤 Using WebSocket for text mode voice processing");
        await this.processVoiceWithStreaming(audioBlob);
      } else {
        console.log("🎤 Fallback to regular API for voice processing");
        await this.processVoiceWithAPI(audioBlob);
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

  async processVoiceWithWebRTC(audioBlob) {
    console.log(
      "🎤 Processing voice with WebRTC, audio blob size:",
      audioBlob.size
    );

    // Convert audio to base64 for WebRTC transmission
    const reader = new FileReader();
    const audioBase64 = await new Promise((resolve, reject) => {
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(audioBlob);
    });

    console.log("🎤 Audio converted to base64, length:", audioBase64.length);

    // Check data channel state
    if (!this.dataChannel) {
      console.error("❌ No data channel available");
      return;
    }

    if (this.dataChannel.readyState !== "open") {
      console.error(
        "❌ Data channel not open, state:",
        this.dataChannel.readyState
      );
      return;
    }

    // Send audio data via WebRTC data channel for ultra-low latency
    const message = {
      type: "voice_chat",
      audio_data: audioBase64,
      session_id: this.sessionId,
    };

    console.log("🎤 Sending voice data via WebRTC data channel:", message.type);
    this.dataChannel.send(JSON.stringify(message));

    console.log("🎤 Voice data sent via WebRTC data channel");
    // The response will be handled by the WebRTC data channel handlers
    // which will stream the LLM tokens and TTS chunks in real-time
  }

  async processVoiceWithStreaming(audioBlob) {
    // Convert audio to base64 for WebSocket transmission
    const reader = new FileReader();
    const audioBase64 = await new Promise((resolve, reject) => {
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = reject;
      reader.readAsDataURL(audioBlob);
    });

    // Send audio data via WebSocket for streaming processing
    // Note: text mode voice input should NOT have automatic TTS
    // Only voice mode realtime conversation should have TTS
    this.websocket.send(
      JSON.stringify({
        type: "voice_chat",
        audio_data: audioBase64,
        session_id: this.sessionId,
        mode: this.currentMode, // "text" or "voice" - tells backend to skip TTS for text mode
      })
    );

    // The response will be handled by the WebSocket message handlers
    // Text mode: only LLM tokens (no TTS)
    // Voice mode: LLM tokens + TTS chunks
  }

  async processVoiceWithAPI(audioBlob) {
    // Fallback to regular API if WebSocket is not available
    const formData = new FormData();
    formData.append("file", audioBlob, "voice_input.webm");

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

  addMessage(content, sender, isStreaming = false) {
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

    // Create a content element with a specific class for streaming
    const contentElement = document.createElement(
      sender === "ai" ? "div" : "p"
    );
    contentElement.className = "leading-relaxed message-content";
    if (sender === "ai") {
      contentElement.innerHTML = this.renderMarkdown(content);
    } else {
      contentElement.textContent = this.escapeHtml(content);
    }
    messageBubble.appendChild(contentElement);

    // Add per-message actions (speaker icon for AI messages)
    if (sender === "ai") {
      const actions = document.createElement("div");
      actions.className = "mt-2 flex items-center space-x-3";

      const speakBtn = document.createElement("button");
      speakBtn.className = "text-gray-400 hover:text-gray-600";
      speakBtn.title = "Read aloud";
      speakBtn.innerHTML = '<i class="fas fa-volume-up text-xs"></i>';
      speakBtn.addEventListener("click", () => {
        try {
          const msgNode = messageBubble.querySelector(".message-content");
          const textToRead = msgNode ? msgNode.textContent : content;
          if (textToRead && textToRead.trim()) {
            this.readAloud(textToRead);
          }
        } catch (e) {
          console.error("Failed to trigger read aloud:", e);
        }
      });

      actions.appendChild(speakBtn);
      messageBubble.appendChild(actions);
    }

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

    return messageDiv;
  }

  renderMarkdown(md) {
    try {
      if (window.marked && window.DOMPurify) {
        const html = marked.parse(md || "");
        return DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
      }
      // Fallback to escaped text if libraries not loaded
      const div = document.createElement("div");
      div.textContent = md || "";
      return div.innerHTML;
    } catch (e) {
      const div = document.createElement("div");
      div.textContent = md || "";
      return div.innerHTML;
    }
  }

  async readAloud(text) {
    try {
      const response = await fetch("/api/tts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        const t = await response.text();
        throw new Error(`TTS failed: ${response.status} ${t}`);
      }
      const result = await response.json();
      if (result.audio_base64) {
        await this.playAudio(result.audio_base64);
      }
    } catch (error) {
      console.error("Read aloud error:", error);
      this.showError("Could not read the response aloud. Please try again.");
    }
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

  updateConnectionStatus(status) {
    if (!this.connectionStatus) return;

    switch (status) {
      case "connected":
        this.connectionStatus.textContent = "🟢 Zevo AI Connected";
        this.connectionStatus.className = "text-sm text-green-500";
        break;
      case "disconnected":
        this.connectionStatus.textContent = "🔴 Zevo AI Disconnected";
        this.connectionStatus.className = "text-sm text-red-500";
        break;
      case "error":
        this.connectionStatus.textContent = "⚠️ Connection Error";
        this.connectionStatus.className = "text-sm text-yellow-500";
        break;
      default:
        this.connectionStatus.textContent = "🟡 Connecting...";
        this.connectionStatus.className = "text-sm text-yellow-500";
    }
  }

  initializeWebSocket() {
    return new Promise((resolve, reject) => {
      try {
        // Use secure WebSocket for HTTPS sites
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/ws/chat/${this.sessionId}`;

        console.log("Attempting WebSocket connection to:", wsUrl);

        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
          console.log("WebSocket connected");
          this.updateConnectionStatus("connected");
          // Clear any existing timeout
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
          }
          resolve(true);
        };

        this.websocket.onmessage = (event) => {
          this.handleWebSocketMessage(event);
        };

        this.websocket.onclose = () => {
          console.log("WebSocket disconnected");
          this.updateConnectionStatus("disconnected");
          // Attempt to reconnect after 3 seconds
          setTimeout(() => {
            if (
              !this.websocket ||
              this.websocket.readyState === WebSocket.CLOSED
            ) {
              this.initializeWebSocket();
            }
          }, 3000);
        };

        this.websocket.onerror = (error) => {
          console.error("WebSocket error:", error);
          console.error("WebSocket readyState:", this.websocket.readyState);
          this.updateConnectionStatus("error");
          reject(error);
        };
      } catch (error) {
        console.error("Failed to initialize WebSocket:", error);
        this.updateConnectionStatus("error");
        reject(error);
      }
    });
  }

  handleWebSocketMessage(event) {
    try {
      const data = JSON.parse(event.data);
      console.log("📨 WebSocket message:", data);

      switch (data.type) {
        case "llm_start":
          this.handleLLMStart(data);
          break;
        case "llm_token":
          this.handleLLMToken(data);
          break;
        case "tts_start":
          this.handleTTSStart(data);
          break;
        case "tts_chunk":
          this.handleTTSChunk(data);
          break;
        case "transcription":
          this.handleTranscription(data);
          break;
        case "complete":
          this.handleStreamingComplete(data);
          break;
        case "error":
          this.handleStreamingError(data);
          break;
        case "pong":
          // Handle ping response
          break;
        default:
          console.log("Unknown message type:", data.type);
      }
    } catch (error) {
      console.error("Error parsing WebSocket message:", error);
    }
  }

  handleLLMStart(data) {
    console.log("LLM generation started");
    this.isStreaming = true;
    this.currentStreamingMessage = this.addMessage("", "ai", true);
    console.log(
      "currentStreamingMessage set to:",
      this.currentStreamingMessage
    );
    this.showTypingIndicator();

    // Stop any ongoing audio
    this.stopAudio();

    // Initialize streaming buffer and debounce timer (Markdown at completion only)
    this.streamingTextBuffer = "";
    this.streamDebounceTimer = null;
    this.streamDebounceMs = 50;
  }

  handleLLMToken(data) {
    console.log("handleLLMToken called with:", data);
    console.log("currentStreamingMessage:", this.currentStreamingMessage);

    // Create message if it doesn't exist (for voice mode)
    if (!this.currentStreamingMessage) {
      console.log("Creating new streaming message for voice mode");
      this.currentStreamingMessage = this.addMessage("", "ai", true);
      this.isStreaming = true;
    }

    if (this.currentStreamingMessage) {
      const messageContent =
        this.currentStreamingMessage.querySelector(".message-content");
      console.log("messageContent element:", messageContent);

      if (messageContent) {
        // Keep streaming plain text to avoid malformed Markdown mid-stream (debounced)
        if (data.full_response) {
          this.streamingTextBuffer = data.full_response;
        } else if (data.token) {
          this.streamingTextBuffer =
            (this.streamingTextBuffer || "") + data.token;
        }
        if (!this.streamDebounceTimer) {
          this.streamDebounceTimer = setTimeout(() => {
            try {
              messageContent.textContent = this.streamingTextBuffer || "";
              this.scrollToBottom();
            } finally {
              this.streamDebounceTimer = null;
            }
          }, this.streamDebounceMs);
        }
      } else {
        console.error("Could not find .message-content element");
      }
    } else {
      console.error("No currentStreamingMessage set");
    }
  }

  handleTTSStart(data) {
    console.log("TTS generation started");
    this.hideTypingIndicator();
  }

  async handleTTSChunk(data) {
    try {
      console.log("🔊 TTS chunk received:", data);
      if (!data.audio_chunk) {
        console.warn("⚠️ TTS chunk has no audio_chunk data");
        return;
      }

      // Ensure AudioContext exists for buffered playback
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext ||
          window.webkitAudioContext)();
        // Resume AudioContext if suspended (required for autoplay policy)
        if (this.audioContext.state === "suspended") {
          await this.audioContext.resume();
        }
      }

      // Decode and queue chunk for scheduled playback to avoid choppy audio
      const buffer = await this.decodeAudioOptimized(
        data.audio_chunk,
        this.audioCodec
      );
      if (!this.audioQueue) this.audioQueue = [];
      if (!this.ttsChunkCount) this.ttsChunkCount = 0;
      
      this.audioQueue.push(buffer);
      this.ttsChunkCount++;
      
      // Build buffer before starting playback (smoother audio)
      // Wait for at least 3 chunks or start immediately if queue is getting full
      const minBufferChunks = 3;
      const shouldStartPlayback = 
        this.ttsChunkCount >= minBufferChunks || 
        this.audioQueue.length >= 5;
      
      if (shouldStartPlayback && !this.isPlayingAudio) {
        console.log(`🔊 Starting TTS playback (buffer: ${this.audioQueue.length} chunks)`);
        this.isPlayingAudio = true;
      }
      
      if (this.isPlayingAudio || shouldStartPlayback) {
        this.scheduleAudioPlayback();
      }
    } catch (error) {
      console.error("Error handling TTS chunk:", error);
    }
  }

  handleTranscription(data) {
    try {
      if (data.text) {
        console.log("🎤 Transcription received:", data.text);
        // Add transcribed text to conversation
        this.addMessage(data.text, "user");

        // Add to conversation history
        this.conversationHistory.push({
          role: "user",
          content: data.text,
          timestamp: new Date().toISOString(),
        });
      }
    } catch (error) {
      console.error("Error handling transcription:", error);
    }
  }

  handleStreamingComplete(data) {
    console.log("✅ Streaming complete - AI response finished");
    this.isStreaming = false;
    this.isProcessing = false;
    this.currentStreamingMessage = null;
    this.hideTypingIndicator();

    // Flush pending debounce update
    if (this.streamDebounceTimer) {
      clearTimeout(this.streamDebounceTimer);
      this.streamDebounceTimer = null;
    }

    // Ensure any remaining audio chunks are scheduled for playback
    if (this.audioQueue && this.audioQueue.length > 0 && !this.isPlayingAudio) {
      console.log(`🔊 Scheduling remaining ${this.audioQueue.length} audio chunks`);
      this.isPlayingAudio = true;
      this.scheduleAudioPlayback();
    }

    // Reset TTS chunk counter for next response
    this.ttsChunkCount = 0;

    // Add to conversation history
    this.conversationHistory.push({
      role: "assistant",
      content: data.response,
      timestamp: new Date().toISOString(),
    });

    // Finalize: render Markdown once full text is available
    try {
      const last = this.chatMessages.lastElementChild;
      if (
        last &&
        last.querySelector &&
        last.querySelector(".message-content")
      ) {
        const el = last.querySelector(".message-content");
        el.innerHTML = this.renderMarkdown(data.response || "");
      }
    } catch (e) {
      console.warn("Markdown finalize render failed:", e);
    }

    // Display latency report if available
    if (this.showLatencyReport && data.latency_report) {
      console.log("📊 Streaming Latency Report:", data.latency_report);
      this.displayLatencyReport(data.latency_report);
    }

    // For voice mode: Wait for audio to finish playing before restarting listening
    if (this.currentMode === "voice" && this.isInCall) {
      console.log(
        "🎤 Voice mode: Waiting for audio to finish before restarting listening"
      );
      // Only schedule restart if no audio is currently playing
      if (!this.audioPlaying) {
        this.scheduleRestartListening();
      } else {
        console.log(
          "🎤 Audio still playing, will restart listening when audio finishes"
        );
      }
    }
  }

  handleStreamingError(data) {
    console.error("Streaming error:", data.message);
    this.isStreaming = false;
    this.isProcessing = false;
    this.currentStreamingMessage = null;
    this.hideTypingIndicator();
    this.showError(data.message);
  }

  // Audio optimization functions
  detectAudioCodec(audioBase64) {
    // Simple detection based on data patterns
    const header = atob(audioBase64.substring(0, 20));
    if (header.includes("OggS")) return "opus";
    if (header.includes("RIFF")) return "wav";
    if (header.includes("ID3") || header.includes("\xff\xfb")) return "mp3";
    return "wav"; // Default fallback
  }

  async decodeAudioOptimized(audioBase64, codec = "wav") {
    try {
      const audioData = atob(audioBase64);
      const audioArray = new Uint8Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
      }

      // Try to decode as WAV first (most compatible)
      try {
        return await this.audioContext.decodeAudioData(audioArray.buffer);
      } catch (wavError) {
        console.log("WAV decoding failed, trying as raw audio data");

        // If WAV decoding fails, try to play as HTML5 audio
        const audioBlob = new Blob([audioArray], { type: "audio/wav" });
        const audioUrl = URL.createObjectURL(audioBlob);

        // Use HTML5 audio for playback
        const audio = new Audio(audioUrl);
        audio
          .play()
          .catch((err) => console.warn("Audio playback failed:", err));

        // Return a mock buffer for compatibility
        return {
          duration: 0.1, // Short duration placeholder
          length: 4410, // 0.1s at 44.1kHz
          getChannelData: () => new Float32Array(4410),
          numberOfChannels: 1,
          sampleRate: 44100,
        };
      }
    } catch (error) {
      console.warn(`Audio decoding failed for ${codec}:`, error);

      // Final fallback: return a silent buffer
      return {
        duration: 0.1,
        length: 4410,
        getChannelData: () => new Float32Array(4410),
        numberOfChannels: 1,
        sampleRate: 44100,
      };
    }
  }

  calculateOptimalBufferSize() {
    // Adaptive buffering based on network conditions
    if (this.networkLatency < 50) {
      return 2; // Low latency, minimal buffering
    } else if (this.networkLatency < 150) {
      return 3; // Medium latency, moderate buffering
    } else {
      return 5; // High latency, more buffering
    }
  }

  // WebRTC functions
  async initializeWebRTC() {
    try {
      console.log("🔗 Initializing WebRTC with optimized settings...");
      this.webrtcConnection = new RTCPeerConnection({
        iceServers: this.iceServers,
        iceCandidatePoolSize: 10, // Pre-gather ICE candidates
        bundlePolicy: "max-bundle", // Optimize for single connection
        rtcpMuxPolicy: "require", // Reduce connection overhead
      });

      // Create data channel for audio streaming
      this.dataChannel = this.webrtcConnection.createDataChannel("audio", {
        ordered: true,
        maxRetransmits: 3,
      });

      // Set up data channel handlers
      this.setupDataChannelHandlers();

      // Handle ICE candidates
      this.webrtcConnection.onicecandidate = (event) => {
        if (event.candidate) {
          // Send ICE candidate to server
          this.sendIceCandidate(event.candidate);
        }
      };

      // Handle connection state changes
      this.webrtcConnection.onconnectionstatechange = () => {
        console.log(
          "WebRTC connection state:",
          this.webrtcConnection.connectionState
        );
        if (this.webrtcConnection.connectionState === "connected") {
          // Don't override main connection status, just set WebRTC flag
          this.useWebRTC = true;
          console.log("✅ WebRTC ready for voice calls");
        } else if (this.webrtcConnection.connectionState === "disconnected") {
          this.useWebRTC = false;
          console.log("⚠️ WebRTC disconnected, falling back to WebSocket");
        }
      };

      // Set connection timeout to prevent delays
      const connectionTimeout = setTimeout(() => {
        if (this.webrtcConnection.connectionState !== "connected") {
          console.warn(
            "⚠️ WebRTC connection timeout, falling back to WebSocket"
          );
          this.useWebRTC = false;
        }
      }, 5000); // 5-second timeout for faster fallback

      return true;
    } catch (error) {
      console.error("WebRTC initialization failed:", error);
      return false;
    }
  }

  setupDataChannelHandlers() {
    if (!this.dataChannel) return;

    // Handle data channel events
    this.dataChannel.onopen = () => {
      console.log("WebRTC data channel opened");
      this.useWebRTC = true;
    };

    this.dataChannel.onmessage = (event) => {
      try {
        console.log("📨 WebRTC data channel message received:", event.data);
        const data = JSON.parse(event.data);
        console.log("📨 Parsed WebRTC message:", data);

        if (data.type === "audio") {
          this.handleWebRTCAudioData(data.content);
        } else if (data.type === "text") {
          this.handleWebRTCTextData(data);
        } else if (data.type === "llm_token") {
          this.handleLLMToken(data);
        } else if (data.type === "tts_chunk") {
          this.handleTTSChunk(data);
        } else if (data.type === "transcription") {
          this.handleTranscription(data);
        } else if (data.type === "complete") {
          this.handleStreamingComplete(data);
        } else if (data.type === "error") {
          this.handleStreamingError(data);
        } else {
          console.log("Unknown WebRTC message type:", data.type);
        }
      } catch (error) {
        console.error("Error parsing WebRTC message:", error);
        // If it's not JSON, treat it as audio data
        this.handleWebRTCAudioData(event.data);
      }
    };

    this.dataChannel.onclose = () => {
      console.log("WebRTC data channel closed");
      this.useWebRTC = false;

      // Attempt to recreate data channel if connection is still active
      if (
        this.webrtcConnection &&
        this.webrtcConnection.connectionState === "connected"
      ) {
        console.log("🔄 Attempting to recreate data channel...");
        setTimeout(() => {
          try {
            this.dataChannel = this.webrtcConnection.createDataChannel(
              "audio",
              {
                ordered: true,
                maxRetransmits: 3,
              }
            );
            this.setupDataChannelHandlers();
          } catch (error) {
            console.error("Failed to recreate data channel:", error);
          }
        }, 1000);
      }
    };
  }

  async createWebRTCOffer() {
    try {
      console.log("🔗 Creating WebRTC offer...");
      const offer = await this.webrtcConnection.createOffer();
      await this.webrtcConnection.setLocalDescription(offer);

      console.log("🔗 Sending WebRTC offer to server...");
      // Send offer to server
      const response = await fetch("/api/webrtc/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: this.sessionId,
          offer: offer,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("🔗 Received WebRTC answer from server");
        await this.webrtcConnection.setRemoteDescription(data.answer);
        console.log("🔗 WebRTC connection established successfully");
        return true;
      } else {
        console.error("❌ WebRTC offer failed:", response.status);
        return false;
      }
    } catch (error) {
      console.error("WebRTC offer creation failed:", error);
      return false;
    }
  }

  async sendIceCandidate(candidate) {
    try {
      await fetch("/api/webrtc/ice-candidate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: this.sessionId,
          candidate: candidate,
        }),
      });
    } catch (error) {
      console.error("Failed to send ICE candidate:", error);
    }
  }

  handleWebRTCAudioData(audioData) {
    // Handle audio data received via WebRTC data channel
    try {
      const audioBase64 = audioData;
      this.playAudioChunk(audioBase64);
    } catch (error) {
      console.error("Error handling WebRTC audio data:", error);
    }
  }

  handleWebRTCTextData(data) {
    try {
      console.log("Received WebRTC text data:", data);
      if (data.content) {
        this.addMessage(data.content, "assistant");
      }
    } catch (error) {
      console.error("Error handling WebRTC text data:", error);
    }
  }

  // Mode Management Functions
  toggleMode() {
    this.currentMode = this.currentMode === "text" ? "voice" : "text";
    this.updateModeUI();
    console.log(`Switched to ${this.currentMode} mode`);
  }

  updateModeUI() {
    if (this.currentMode === "text") {
      // Text mode
      this.modeToggleThumb.style.transform = "translateX(0.25rem)";
      this.modeToggle.style.backgroundColor = "#e5e7eb";
      this.textInputArea.style.display = "block";
      this.voiceInputArea.style.display = "none";

      // Update welcome message for text mode
      this.updateWelcomeMessage("text");

      // Stop any ongoing voice mode
      if (this.isVoiceModeActive) {
        this.stopVoiceMode();
      }

      // Focus on text input
      setTimeout(() => {
        this.messageInput.focus();
      }, 100);
    } else {
      // Voice mode
      this.modeToggleThumb.style.transform = "translateX(1.75rem)";
      this.modeToggle.style.backgroundColor = "#3b82f6";
      this.textInputArea.style.display = "none";
      this.voiceInputArea.style.display = "block";

      // Update welcome message for voice mode
      this.updateWelcomeMessage("voice");

      // Stop any ongoing text recording
      if (this.isRecording) {
        this.stopRecording();
      }
    }
  }

  updateWelcomeMessage(mode) {
    const welcomeMessage = document.querySelector(
      "#chatMessages .flex.items-start.space-x-3"
    );
    if (welcomeMessage) {
      const messageText = welcomeMessage.querySelector("p");
      if (messageText) {
        if (mode === "text") {
          messageText.textContent =
            "Hello! I'm Zevo AI, your conversational assistant. You can type your message below or use the voice button for quick voice input. How can I help you today?";
        } else {
          messageText.textContent =
            "Hello! I'm Zevo AI, your voice assistant. Tap the microphone button to start a natural voice conversation. I'll listen and respond just like talking to a person!";
        }
      }
    }
  }

  async toggleVoiceMode() {
    if (this.isInCall) {
      this.endCall();
    } else {
      await this.startCall();
    }
  }

  async startCall() {
    this.isInCall = true;
    this.callStartTime = Date.now();
    this.continuousListening = true;

    // Pause health checks during active call for better performance
    this.pauseHealthChecks();

    // Initialize WebRTC for voice calls (phone-like conversation)
    if (!this.webrtcConnection) {
      console.log("🔗 Initializing WebRTC for voice call");
      await this.initializeWebRTC();
    }

    // Show active call interface
    if (this.startCallInterface) {
      this.startCallInterface.style.display = "none";
    }
    if (this.activeCallInterface) {
      this.activeCallInterface.style.display = "flex";
    }

    // Start call duration timer
    this.startCallDurationTimer();

    // Start continuous listening
    this.startContinuousListening();

    // Update call status
    this.updateCallStatus("Connected", "green");
    if (this.callInfo) {
      this.callInfo.textContent =
        "Two-way conversation active - speak naturally";
    }

    console.log("Call started - continuous conversation mode with WebRTC");
  }

  endCall() {
    console.log("📞 Ending call - stopping all conversation processes");

    // Stop all conversation processes immediately
    this.isInCall = false;
    this.continuousListening = false;
    this.isProcessing = false;
    this.isStreaming = false;
    this.audioPlaying = false;
    this.callStartTime = null;

    // Resume health checks after call ends
    this.resumeHealthChecks();

    // Stop call duration timer
    if (this.callDurationInterval) {
      clearInterval(this.callDurationInterval);
      this.callDurationInterval = null;
    }

    // Stop any ongoing recording immediately
    if (this.isRecording) {
      console.log("📞 Stopping ongoing recording");
      this.stopRecording();
    }

    // Stop all audio playback
    this.stopAudio();

    // Clean up ALL intervals and timeouts
    if (this.voiceModeInterval) {
      clearInterval(this.voiceModeInterval);
      this.voiceModeInterval = null;
    }

    if (this.conversationInterval) {
      clearInterval(this.conversationInterval);
      this.conversationInterval = null;
    }

    if (this.voiceModeInterruptionTimeout) {
      clearTimeout(this.voiceModeInterruptionTimeout);
      this.voiceModeInterruptionTimeout = null;
    }

    if (this.voiceProcessingInterval) {
      clearInterval(this.voiceProcessingInterval);
      this.voiceProcessingInterval = null;
    }

    if (this.restartListeningTimeout) {
      clearTimeout(this.restartListeningTimeout);
      this.restartListeningTimeout = null;
    }

    // Clear any pending streaming messages
    this.currentStreamingMessage = null;

    // Show start call interface
    if (this.startCallInterface) {
      this.startCallInterface.style.display = "flex";
    }
    if (this.activeCallInterface) {
      this.activeCallInterface.style.display = "none";
    }

    // Reset all call state
    this.isMuted = false;
    this.isOnHold = false;

    // Update call status
    this.updateCallStatus("Call ended", "gray");

    console.log("📞 Call ended - all processes stopped");
  }

  stopVoiceMode() {
    // This function is now handled by endCall()
    this.endCall();
  }

  startContinuousVoiceRecording() {
    // Start recording with VAD for continuous conversation
    this.startRecording();

    // Set up continuous monitoring
    this.voiceModeInterval = setInterval(() => {
      if (this.isVoiceModeActive && !this.isRecording && !this.isProcessing) {
        // Auto-restart recording if not currently recording
        this.startRecording();
      }
    }, 1000);

    // Set up interruption handling for voice mode
    this.setupVoiceModeInterruptionHandling();
  }

  setupVoiceModeInterruptionHandling() {
    // Enhanced interruption handling for voice mode
    this.voiceModeInterruptionTimeout = setTimeout(() => {
      if (this.isVoiceModeActive && this.isProcessing) {
        // Send interruption signal to backend
        this.sendInterruptionSignal();
      }
    }, 500); // Quick interruption detection
  }

  sendInterruptionSignal() {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(
        JSON.stringify({
          type: "interruption",
          session_id: this.sessionId,
          timestamp: Date.now(),
          mode: "voice",
        })
      );
      console.log("Sent interruption signal for voice mode");
    }
  }

  toggleMute() {
    // Toggle audio output mute
    if (this.audioPlayer.muted) {
      this.audioPlayer.muted = false;
      this.muteBtn.innerHTML = '<i class="fas fa-volume-up text-lg"></i>';
      this.muteBtn.title = "Mute";
    } else {
      this.audioPlayer.muted = true;
      this.muteBtn.innerHTML = '<i class="fas fa-volume-mute text-lg"></i>';
      this.muteBtn.title = "Unmute";
    }
  }

  // Call Control Functions
  startCallDurationTimer() {
    this.callDurationInterval = setInterval(() => {
      if (this.isInCall && this.callStartTime && this.callDuration) {
        const duration = Date.now() - this.callStartTime;
        const minutes = Math.floor(duration / 60000);
        const seconds = Math.floor((duration % 60000) / 1000);
        this.callDuration.textContent = `${minutes
          .toString()
          .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
      }
    }, 1000);
  }

  startContinuousListening() {
    console.log("🎤 Starting continuous listening for voice call");

    // Start continuous recording with VAD for two-way conversation
    this.startRecording();

    // Set up continuous monitoring for call mode (less aggressive)
    this.voiceModeInterval = setInterval(() => {
      // Only continue if call is still active
      if (!this.isInCall) {
        console.log("📞 Call ended - stopping continuous listening monitoring");
        return;
      }

      if (
        this.isInCall &&
        this.continuousListening &&
        !this.isRecording &&
        !this.isProcessing &&
        !this.isStreaming && // Don't restart while AI is responding
        !this.isMuted &&
        !this.isOnHold
      ) {
        console.log("🔄 Auto-restarting recording for continuous conversation");
        // Auto-restart recording if not currently recording
        this.startRecording();
      }
    }, 5000); // Check every 5 seconds to reduce aggressive listening

    // Set up two-way conversation monitoring
    this.setupTwoWayConversation();

    // Set up automatic voice processing with VAD
    this.setupVoiceActivityDetection();
  }

  setupTwoWayConversation() {
    // Monitor for AI responses and automatically restart listening
    this.conversationInterval = setInterval(() => {
      // Only continue if call is still active
      if (!this.isInCall) {
        console.log("📞 Call ended - stopping two-way conversation monitoring");
        return;
      }

      if (
        this.isInCall &&
        !this.isProcessing &&
        !this.isMuted &&
        !this.isOnHold &&
        !this.isStreaming // Don't restart while AI is still responding
      ) {
        // Check if we should restart listening after AI response
        if (!this.isRecording && this.continuousListening) {
          console.log(
            "🔄 Two-way conversation: Restarting listening after AI response"
          );
          this.startRecording();
        }
      }
    }, 4000); // Check every 4 seconds to avoid too aggressive restarting
  }

  setupVoiceActivityDetection() {
    // Set up automatic voice processing with VAD
    if (this.vad) {
      console.log(
        "🎤 Setting up Voice Activity Detection for continuous conversation"
      );

      // Process voice automatically when user stops speaking
      this.vad.on("voice_start", () => {
        console.log("🎤 Voice detected - user started speaking");
      });

      this.vad.on("voice_stop", () => {
        console.log("🎤 Voice stopped - processing speech automatically");
        if (
          this.isRecording &&
          this.isInCall &&
          !this.isMuted &&
          !this.isOnHold
        ) {
          // Stop recording and process voice automatically
          this.stopRecording();
        }
      });
    } else {
      console.log("⚠️ VAD not available, using timer-based voice processing");

      // Fallback: Use timer-based voice processing
      this.voiceProcessingInterval = setInterval(() => {
        if (
          this.isRecording &&
          this.isInCall &&
          !this.isMuted &&
          !this.isOnHold
        ) {
          // Stop recording after 3 seconds of continuous recording
          console.log("🎤 Timer-based voice processing - stopping recording");
          this.stopRecording();
        }
      }, 5000); // Process voice every 5 seconds to reduce automatic chatting
    }
  }

  scheduleRestartListening() {
    // Wait for audio to finish playing before restarting listening
    console.log("🎤 Scheduling restart of listening after audio finishes");

    // Clear any existing restart timeout
    if (this.restartListeningTimeout) {
      clearTimeout(this.restartListeningTimeout);
    }

    // Wait for audio to finish playing
    this.restartListeningTimeout = setTimeout(() => {
      // Only restart if call is still active
      if (!this.isInCall) {
        console.log("📞 Call ended - cancelling restart listening");
        return;
      }

      if (
        this.isInCall &&
        !this.isRecording &&
        !this.isProcessing &&
        !this.isStreaming &&
        !this.isMuted &&
        !this.isOnHold &&
        !this.audioPlaying
      ) {
        console.log(
          "🎤 Restarting listening after AI response completed and audio finished"
        );
        this.startRecording();
      } else if (this.audioPlaying) {
        console.log("🎤 Audio still playing, waiting for completion...");
        // Reschedule if audio is still playing
        this.scheduleRestartListening();
      }
    }, 1000); // Check every 1 second
  }

  pauseHealthChecks() {
    // Pause health checks during active call for better performance
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
      console.log("⏸️ Health checks paused during call");
    }
  }

  resumeHealthChecks() {
    // Resume health checks after call ends
    if (!this.healthCheckInterval) {
      this.healthCheckInterval = setInterval(() => {
        this.checkConnection();
      }, 300000); // Check every 5 minutes to reduce log spam
      console.log("▶️ Health checks resumed after call");
    }
  }

  updateCallStatus(status, color) {
    if (this.callStatusText) {
      this.callStatusText.textContent = status;
    }
    if (this.callStatusIndicator) {
      this.callStatusIndicator.className = `w-3 h-3 rounded-full ${
        color === "green" ? "bg-green-500 animate-pulse" : "bg-gray-500"
      }`;
    }
  }

  toggleCallMute() {
    this.isMuted = !this.isMuted;

    if (this.isMuted) {
      this.callMuteBtn.className =
        "w-16 h-16 bg-red-100 hover:bg-red-200 text-red-600 rounded-full flex items-center justify-center transition-all";
      this.callMuteBtn.innerHTML =
        '<i class="fas fa-microphone-slash text-xl"></i>';
      this.callMuteBtn.title = "Unmute";

      // Stop recording if muted, but keep conversation active
      if (this.isRecording) {
        this.stopRecording();
      }

      this.updateCallStatus("Muted", "red");
      if (this.callInfo) {
        this.callInfo.textContent = "Microphone muted - AI can still speak";
      }

      console.log("🔇 Call muted - AI can still respond");
    } else {
      this.callMuteBtn.className =
        "w-16 h-16 bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-full flex items-center justify-center transition-all";
      this.callMuteBtn.innerHTML = '<i class="fas fa-microphone text-xl"></i>';
      this.callMuteBtn.title = "Mute";

      this.updateCallStatus("Connected", "green");
      if (this.callInfo) {
        this.callInfo.textContent =
          "Two-way conversation active - speak naturally";
      }

      // Restart recording if not on hold
      if (!this.isOnHold) {
        this.startRecording();
      }

      console.log("🔊 Call unmuted - full conversation active");
    }

    console.log(`Call ${this.isMuted ? "muted" : "unmuted"}`);
  }

  toggleCallHold() {
    this.isOnHold = !this.isOnHold;

    if (this.isOnHold) {
      this.callHoldBtn.className =
        "w-16 h-16 bg-yellow-200 hover:bg-yellow-300 text-yellow-700 rounded-full flex items-center justify-center transition-all";
      this.callHoldBtn.innerHTML = '<i class="fas fa-play text-xl"></i>';
      this.callHoldBtn.title = "Resume";

      // Stop recording when on hold
      if (this.isRecording) {
        this.stopRecording();
      }

      this.updateCallStatus("On Hold", "yellow");
      if (this.callInfo) {
        this.callInfo.textContent = "Call on hold";
      }
    } else {
      this.callHoldBtn.className =
        "w-16 h-16 bg-yellow-100 hover:bg-yellow-200 text-yellow-600 rounded-full flex items-center justify-center transition-all";
      this.callHoldBtn.innerHTML = '<i class="fas fa-pause text-xl"></i>';
      this.callHoldBtn.title = "Hold";

      this.updateCallStatus("Connected", "green");
      if (this.callInfo) {
        this.callInfo.textContent =
          "Two-way conversation active - speak naturally";
      }

      // Resume recording if not muted
      if (!this.isMuted) {
        this.startRecording();
      }
    }

    console.log(`Call ${this.isOnHold ? "on hold" : "resumed"}`);
  }

  handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + Enter to send message in text mode
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      if (this.currentMode === "text" && this.messageInput.value.trim()) {
        e.preventDefault();
        this.sendMessage();
      }
    }

    // Space bar to toggle voice mode (when not in text input)
    if (e.key === " " && e.target !== this.messageInput) {
      e.preventDefault();
      if (this.currentMode === "voice") {
        this.toggleVoiceMode();
      }
    }

    // Escape to end call
    if (e.key === "Escape") {
      if (this.currentMode === "voice" && this.isInCall) {
        this.endCall();
      }
    }

    // M key to toggle mute
    if (e.key === "m" || e.key === "M") {
      if (this.currentMode === "voice" && this.isInCall) {
        e.preventDefault();
        this.toggleCallMute();
      }
    }

    // H key to toggle hold
    if (e.key === "h" || e.key === "H") {
      if (this.currentMode === "voice" && this.isInCall) {
        e.preventDefault();
        this.toggleCallHold();
      }
    }

    // T key to toggle between text and voice modes
    if (e.key === "t" || e.key === "T") {
      if (e.target !== this.messageInput) {
        e.preventDefault();
        this.toggleMode();
      }
    }
  }

  async sendWebRTCMessage(message) {
    if (this.dataChannel && this.dataChannel.readyState === "open") {
      this.dataChannel.send(
        JSON.stringify({
          type: "message",
          content: message,
          session_id: this.sessionId,
        })
      );
      return true;
    }
    return false;
  }

  // Voice Activity Detection functions
  async initializeVAD() {
    try {
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext ||
          window.webkitAudioContext)();
      }

      // Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      this.microphone = this.audioContext.createMediaStreamSource(stream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 256;
      this.analyser.smoothingTimeConstant = 0.8;

      this.microphone.connect(this.analyser);
      this.audioDataArray = new Uint8Array(this.analyser.frequencyBinCount);

      // Start VAD monitoring
      this.startVADMonitoring();

      return true;
    } catch (error) {
      console.error("VAD initialization failed:", error);
      return false;
    }
  }

  startVADMonitoring() {
    if (!this.analyser || !this.audioDataArray) return;

    const checkVoiceActivity = () => {
      if (!this.vadEnabled) return;

      this.analyser.getByteFrequencyData(this.audioDataArray);

      // Calculate average volume
      const average =
        this.audioDataArray.reduce((sum, value) => sum + value, 0) /
        this.audioDataArray.length;
      const normalizedVolume = average / 255;

      const wasSpeaking = this.isUserSpeaking;
      this.isUserSpeaking = normalizedVolume > this.silenceThreshold;

      // Handle speech state changes
      if (this.isUserSpeaking && !wasSpeaking) {
        this.onSpeechStart();
      } else if (!this.isUserSpeaking && wasSpeaking) {
        this.onSpeechEnd();
      }

      // Continue monitoring
      if (this.vadEnabled) {
        requestAnimationFrame(checkVoiceActivity);
      }
    };

    checkVoiceActivity();
  }

  onSpeechStart() {
    console.log("Speech detected - user started speaking");

    // Stop current TTS if playing
    if (this.isPlayingAudio) {
      this.stopAudio();
      console.log("Stopped TTS due to user speech");
    }

    // Update UI to show user is speaking
    this.updateVoiceIndicator(true);
  }

  onSpeechEnd() {
    console.log("Speech ended - user stopped speaking");

    // Update UI to show user stopped speaking
    this.updateVoiceIndicator(false);

    // Could trigger automatic voice recording here
    // this.startVoiceRecording();
  }

  updateVoiceIndicator(isSpeaking) {
    // Update UI elements to show voice activity
    const voiceIndicator = document.getElementById("voice-indicator");
    if (voiceIndicator) {
      voiceIndicator.textContent = isSpeaking ? "🎤 Speaking..." : "🎤 Ready";
      voiceIndicator.className = isSpeaking ? "voice-active" : "voice-inactive";
    }
  }

  // Interruption handling
  handleUserInterruption() {
    if (this.isPlayingAudio) {
      console.log("User interruption detected - stopping audio");
      this.stopAudio();

      // Send interruption signal to server
      if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
        this.websocket.send(
          JSON.stringify({
            type: "interruption",
            session_id: this.sessionId,
            timestamp: Date.now(),
          })
        );
      }
    }
  }

  async playAudioChunk(audioBase64) {
    try {
      // Buffered playback using WebAudio to prevent choppiness
      if (!this.audioContext) {
        this.audioContext = new (window.AudioContext ||
          window.webkitAudioContext)();
      }
      const buffer = await this.decodeAudioOptimized(
        audioBase64,
        this.audioCodec
      );
      if (!this.audioQueue) this.audioQueue = [];
      this.audioQueue.push(buffer);
      this.scheduleAudioPlayback();
    } catch (error) {
      console.error("Error playing audio chunk:", error);

      // Reset audio playing state on error
      if (this.currentMode === "voice" && this.isInCall) {
        this.audioPlaying = false;
      }
    }
  }

  async scheduleAudioPlayback() {
    if (this.audioQueue.length === 0 || this.isScheduling) return;
    
    // Ensure AudioContext is ready
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
    }
    
    // Resume AudioContext if suspended (required for autoplay policy)
    if (this.audioContext.state === "suspended") {
      try {
        await this.audioContext.resume();
      } catch (error) {
        console.warn("Failed to resume AudioContext:", error);
      }
    }

    this.isScheduling = true;
    const buffer = this.audioQueue.shift();
    const source = this.audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(this.audioContext.destination);

    // Schedule playback timing to avoid gaps
    if (
      !this.nextPlayTime ||
      this.nextPlayTime < this.audioContext.currentTime
    ) {
      // Start immediately if no previous playback or if we're behind
      this.nextPlayTime = this.audioContext.currentTime + 0.01;
    }

    try {
      source.start(this.nextPlayTime);
      this.nextPlayTime += buffer.duration;

      source.onended = () => {
        this.isScheduling = false;

        // Always schedule next chunk if available (smoother playback)
        if (this.audioQueue.length > 0) {
          this.scheduleAudioPlayback();
        } else {
          // No more chunks, mark as done
          this.isPlayingAudio = false;
          console.log("🔊 TTS playback completed");
        }
      };

      this.scheduledSources.push(source);
      this.isPlayingAudio = true;
    } catch (error) {
      console.error("Error starting audio source:", error);
      this.isScheduling = false;
      // Try to schedule next chunk if available
      if (this.audioQueue.length > 0) {
        this.scheduleAudioPlayback();
      } else {
        this.isPlayingAudio = false;
      }
    }
  }

  stopAudio() {
    try {
      // Ensure scheduledSources exists
      if (this.scheduledSources && Array.isArray(this.scheduledSources)) {
        this.scheduledSources.forEach((source) => {
          try {
            source.stop();
          } catch (e) {}
        });
        this.scheduledSources = [];
      }

      // Ensure audioQueue exists
      if (!this.audioQueue || !Array.isArray(this.audioQueue)) {
        this.audioQueue = [];
      } else {
        this.audioQueue = [];
      }

      this.isScheduling = false;
      this.isPlayingAudio = false;

      if (this.audioContext) {
        this.nextPlayTime = this.audioContext.currentTime + 0.05;
      } else {
        this.nextPlayTime = 0;
      }
    } catch (error) {
      console.error("Error stopping audio:", error);
    }
  }

  sendStreamingMessage(message) {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      console.error("WebSocket not connected");
      this.showError("Connection lost. Please refresh the page.");
      return;
    }

    if (this.isProcessing || this.isStreaming) {
      return;
    }

    this.isProcessing = true;

    // Add user message to conversation
    this.conversationHistory.push({
      role: "user",
      content: message,
      timestamp: new Date().toISOString(),
    });

    // Display user message
    this.addMessage(message, "user");

    // Send message via WebSocket
    this.websocket.send(
      JSON.stringify({
        type: "chat",
        message: message,
      })
    );
  }

  displayLatencyReport(latencyReport) {
    // Create latency display element
    const latencyDiv = document.createElement("div");
    latencyDiv.className =
      "latency-report bg-blue-50 border border-blue-200 rounded-lg p-3 mt-2 text-xs";

    const totalTime = latencyReport.total_duration_ms;
    const bottleneck = latencyReport.bottleneck;

    let latencyHtml = `
      <div class="flex items-center justify-between mb-2">
        <span class="font-semibold text-blue-800">📊 Performance Report</span>
        <span class="text-blue-600">Total: ${totalTime}ms</span>
      </div>
      <div class="space-y-1">
    `;

    // Add each step
    for (const [stepName, stepData] of Object.entries(latencyReport.steps)) {
      const percentage = latencyReport.step_percentages[stepName];
      const isBottleneck = stepName === bottleneck;
      const stepClass = isBottleneck
        ? "bg-red-100 text-red-800"
        : "bg-gray-100 text-gray-700";

      latencyHtml += `
        <div class="flex items-center justify-between ${stepClass} px-2 py-1 rounded">
          <span class="capitalize">${stepName.replace("_", " ")}</span>
          <span>${stepData.duration_ms}ms (${percentage}%)</span>
        </div>
      `;
    }

    latencyHtml += `
      </div>
      ${
        bottleneck
          ? `<div class="mt-2 text-red-600 text-xs">⚠️ Bottleneck: ${bottleneck.replace(
              "_",
              " "
            )}</div>`
          : ""
      }
    `;

    latencyDiv.innerHTML = latencyHtml;

    // Add to the last AI message
    const lastMessage = this.chatMessages.lastElementChild;
    if (lastMessage && lastMessage.querySelector(".message-content")) {
      const messageContent = lastMessage.querySelector(".message-content");
      messageContent.appendChild(latencyDiv);
    }
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
