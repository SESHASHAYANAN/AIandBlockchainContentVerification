import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";

const useModelLoader = () => {
  const [model, setModel] = useState(null);
  const [modelError, setModelError] = useState(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        await tf.setBackend("webgl");
        console.log("Using WebGL backend");
      } catch (error) {
        console.warn("WebGL backend failed, falling back to WASM:", error);
        try {
          await tf.setBackend("wasm");
          console.log("Using WASM backend");
        } catch (warnError) {
          const error = "Failed to initialize TensorFlow backend";
          console.error(error, warnError);
          setModelError(error);
          return;
        }
      }

      try {
        await tf.ready();
        console.log("TensorFlow backend initialized:", tf.getBackend());
        const modelLoadPromise = tf.loadLayersModel(
          "/path/to/model/model.json"
        );
        const timeoutPromise = new Promise((_, reject) =>
          setTimeout(() => reject(new Error("Model loading timed out")), 30000)
        );

        const loadedModel = await Promise.race([
          modelLoadPromise,
          timeoutPromise,
        ]);

        const dummyInput = tf.zeros([1, 224, 224, 3]);
        await loadedModel.predict(dummyInput).data();
        dummyInput.dispose();

        setModel(loadedModel);
        console.log("Model loaded successfully");
      } catch (error) {
        const errorMessage = `Failed to load model: ${error.message}`;
        console.error(errorMessage);
        setModelError(errorMessage);
      }
    };

    loadModel();

    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, []);

  return { model, modelError };
};

const preprocessImage = async (imageElement) => {
  return tf.tidy(() => {
    const tensor = tf.browser
      .fromPixels(imageElement)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(255.0)
      .expandDims();
    return tensor;
  });
};

function App() {
  const [walletId, setWalletId] = useState("");
  const [account, setAccount] = useState(null);
  const [verificationMode, setVerificationMode] = useState("image");
  const [newsText, setNewsText] = useState("");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const { model, modelError } = useModelLoader();
  const fileInputRef = useRef(null);

  const validateWallet = () => {
    if (walletId.trim()) {
      setAccount({ accountId: walletId });
    }
  };

  const signIn = () => {
    console.log("Connecting to NEAR wallet...");
  };

  const signOut = () => {
    setAccount(null);
    setWalletId("");
  };

  const getWordCount = () => newsText.trim().split(/\s+/).length;

  const verifyNews = async () => {
    setLoading(true);
    try {
      if (getWordCount() > 150) {
        throw new Error("News content exceeds the 150-word limit.");
      }

      const response = await fetch(
        `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(
          newsText
        )}`
      );

      if (!response.ok) {
        throw new Error(`Wikipedia API error: ${response.statusText}`);
      }

      const data = await response.json();
      setResult({
        isAuthentic: !!data.extract,
        realNews: data.extract || "Not found in Wikipedia.",
        confidence: data.extract ? 80 : 50,
      });
    } catch (error) {
      console.error("News verification failed:", error);
      setResult({
        isAuthentic: false,
        realNews: `Verification failed: ${error.message}`,
        confidence: 0,
      });
    } finally {
      setLoading(false);
    }
  };

  // Image handling functions
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (event) => setImage(event.target.result);
      reader.readAsDataURL(file);
    } else {
      alert("Please upload a valid image file.");
    }
  };

  const detectFakeImage = async () => {
    if (!image || !model) {
      alert("Please upload an image and ensure the model is loaded.");
      return;
    }

    setLoading(true);
    try {
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = image;

      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = (error) =>
          reject(new Error(`Failed to load image: ${error}`));
      });

      const tensor = await preprocessImage(img);
      const predictions = await model.predict(tensor);
      const resultValue = await predictions.data();

      tensor.dispose();
      predictions.dispose();

      setResult({
        isAuthentic: resultValue[0] < 0.5,
        confidence: Math.round(Math.abs(1 - resultValue[0]) * 100),
      });
    } catch (error) {
      console.error("Image verification failed:", error);
      setResult({
        isAuthentic: false,
        confidence: 0,
        error: error.message,
      });
    } finally {
      setLoading(false);
    }
  };

  const styles = {
    app: {
      maxWidth: "800px",
      margin: "0 auto",
      padding: "20px",
      fontFamily: "Arial, sans-serif",
    },
    header: {
      textAlign: "center",
      marginBottom: "30px",
      fontFamily: "Arial, sans-serif",
    },
    walletSection: {
      marginBottom: "20px",
      padding: "15px",
      border: "1px solid #ddd",
      borderRadius: "5px",
    },
    button: {
      padding: "10px 20px",
      margin: "5px",
      borderRadius: "5px",
      border: "none",
      cursor: "pointer",
      backgroundColor: "#007bff",
      color: "white",
    },
    disabledButton: {
      backgroundColor: "#ccc",
      cursor: "not-allowed",
    },
    input: {
      padding: "10px",
      margin: "5px",
      borderRadius: "5px",
      border: "1px solid #ddd",
      width: "100%",
      maxWidth: "300px",
    },
    textarea: {
      width: "100%",
      minHeight: "150px",
      padding: "10px",
      margin: "10px 0",
      borderRadius: "5px",
      border: "1px solid #ddd",
    },
    resultSection: {
      marginTop: "20px",
      padding: "15px",
      borderRadius: "5px",
      backgroundColor: "#f8f9fa",
    },
    authentic: {
      border: "2px solid #28a745",
    },
    suspicious: {
      border: "2px solid #dc3545",
    },
  };

  return (
    <div style={styles.app}>
      <header style={styles.header}>
        <h1>AI CONTENT VERIFICATION USING BLOCKCHAIN</h1>
        <p>
          Verify the authenticity of images and news using AI and blockchain
          technology
        </p>
        {modelError && (
          <div style={{ color: "red", marginTop: "10px" }}>
            Model Error: {modelError}
          </div>
        )}
      </header>

      <div style={styles.walletSection}>
        {!account ? (
          <div>
            <input
              type="text"
              style={styles.input}
              placeholder="Enter your NEAR wallet ID"
              value={walletId}
              onChange={(e) => setWalletId(e.target.value)}
            />
            <button
              style={{
                ...styles.button,
                ...(walletId.trim() ? {} : styles.disabledButton),
              }}
              onClick={validateWallet}
              disabled={!walletId.trim()}
            >
              Validate Wallet
            </button>
            <button
              style={{
                ...styles.button,
                ...(walletId.trim() ? {} : styles.disabledButton),
              }}
              onClick={signIn}
              disabled={!walletId.trim()}
            >
              Connect Wallet
            </button>
          </div>
        ) : (
          <div>
            <span>Connected: {account.accountId}</span>
            <button style={styles.button} onClick={signOut}>
              Disconnect
            </button>
          </div>
        )}
      </div>

      <div style={{ marginBottom: "20px" }}>
        <button
          style={{
            ...styles.button,
            backgroundColor:
              verificationMode === "image" ? "#0056b3" : "#007bff",
          }}
          onClick={() => setVerificationMode("image")}
        >
          Image Verification
        </button>
        <button
          style={{
            ...styles.button,
            backgroundColor:
              verificationMode === "news" ? "#0056b3" : "#007bff",
          }}
          onClick={() => setVerificationMode("news")}
        >
          News Verification
        </button>
      </div>

      {verificationMode === "image" ? (
        <div>
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleImageUpload}
            style={{ display: "none" }}
          />
          <button
            style={styles.button}
            onClick={() => fileInputRef.current.click()}
          >
            Choose Image
          </button>
          {image && (
            <div style={{ marginTop: "20px", textAlign: "center" }}>
              <img src={image} alt="Preview" style={{ maxWidth: "300px" }} />
              <button
                style={{
                  ...styles.button,
                  display: "block",
                  margin: "10px auto",
                }}
                onClick={detectFakeImage}
                disabled={loading}
              >
                {loading ? "Verifying..." : "Verify Image"}
              </button>
            </div>
          )}
        </div>
      ) : (
        <div>
          <textarea
            style={styles.textarea}
            placeholder="Paste news content here (max 150 words)"
            value={newsText}
            onChange={(e) => setNewsText(e.target.value)}
          />
          <div>Words: {getWordCount()}/150</div>
          <button
            style={{
              ...styles.button,
              ...(loading || getWordCount() > 150 ? styles.disabledButton : {}),
            }}
            onClick={verifyNews}
            disabled={loading || getWordCount() > 150}
          >
            {loading ? "Verifying..." : "Verify News"}
          </button>
        </div>
      )}

      {result && (
        <div
          style={{
            ...styles.resultSection,
            ...(result.isAuthentic ? styles.authentic : styles.suspicious),
          }}
        >
          <h3 style={{ color: result.isAuthentic ? "#28a745" : "#dc3545" }}>
            {result.isAuthentic
              ? "✓ Authentic Content"
              : "⚠ Suspicious Content"}
          </h3>
          <div>Confidence: {result.confidence}%</div>
          {result.error && (
            <div style={{ color: "red" }}>Error: {result.error}</div>
          )}
          {verificationMode === "news" && result.realNews && (
            <div style={{ marginTop: "10px" }}>
              <h4>Verified Content:</h4>
              <p>{result.realNews}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
