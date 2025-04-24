 // Game variables
 let gameActive = false;
 let gameLevel = 1;
 let currentText = '';
 let typedText = '';
 let currentIndex = 0;
 let startTime = 0;
 let errors = 0;
 let combo = 0;
 let wpm = 0;
 let accuracy = 100;
 let progressHistory = [];
 let username = '';
 
 // DOM Elements
 const textDisplay = document.getElementById('text-display');
 const textInput = document.getElementById('text-input');
 const wpmDisplay = document.getElementById('wpm');
 const accuracyDisplay = document.getElementById('accuracy');
 const errorsDisplay = document.getElementById('errors');
 const levelDisplay = document.getElementById('level');
 const comboFill = document.getElementById('combo-fill');
 const comboValue = document.getElementById('combo-value');
 const feedbackDisplay = document.getElementById('feedback');
 const startBtn = document.getElementById('start-btn');
 const restartBtn = document.getElementById('restart-btn');
 const settingsBtn = document.getElementById('settings-btn');
 const leaderboardBtn = document.getElementById('leaderboard-btn');
 const gameOverScreen = document.getElementById('game-over');
 const resultWpm = document.getElementById('result-wpm');
 const resultAccuracy = document.getElementById('result-accuracy');
 const resultTime = document.getElementById('result-time');
 const resultLevel = document.getElementById('result-level');
 const progressChart = document.getElementById('progress-chart');
 const tipBox = document.getElementById('tip-box');
 const nextLevelBtn = document.getElementById('next-level-btn');
 const retryBtn = document.getElementById('retry-btn');
 const viewStatsBtn = document.getElementById('view-stats-btn');
 const settingsModal = document.getElementById('settings-modal');
 const closeSettings = document.getElementById('close-settings');
 const saveSettings = document.getElementById('save-settings');
 const leaderboardModal = document.getElementById('leaderboard-modal');
 const closeLeaderboard = document.getElementById('close-leaderboard');
 const difficultySelect = document.getElementById('difficulty');
 const modeSelect = document.getElementById('mode');
 const soundEffectsCheck = document.getElementById('sound-effects');
 const visualEffectsCheck = document.getElementById('visual-effects');
 const leaderboardFilter = document.getElementById('leaderboard-filter');
 const leaderboardBody = document.getElementById('leaderboard-body');
 const missionPrompt = document.getElementById('mission-prompt');
 
 // Game text content
 const textContent = {
     words: {
         easy: [
             "security password firewall encryption algorithm authentication protocol",
             "network database server client system software hardware interface code",
             "access denial terminal command keyboard monitor screen display digital"
         ],
         medium: [
             "encryption algorithms utilize complex mathematical functions to protect sensitive information from unauthorized access",
             "cybersecurity professionals continuously monitor network traffic for signs of intrusion or suspicious activities",
             "two-factor authentication adds an additional layer of security beyond traditional password protection systems"
         ],
         hard: [
             "The implementation of quantum cryptography may render current encryption methods obsolete in the near future.",
             "Zero-day vulnerabilities represent a significant threat as they can be exploited before developers have the opportunity to create patches.",
             "Sophisticated social engineering attacks often bypass traditional security measures by manipulating human psychology rather than technology."
         ],
         expert: [
             "As we navigate the increasingly complex digital landscape, cybersecurity professionals must anticipate zero-day vulnerabilities while simultaneously developing proactive defense mechanisms that adapt to emerging threats in real-time.",
             "The proliferation of Internet of Things (IoT) devices has exponentially increased potential attack vectors, creating unprecedented challenges for network administrators implementing effective security protocols across heterogeneous systems.",
             "Quantum computing advancements threaten to undermine conventional encryption algorithms, necessitating the urgent development and implementation of post-quantum cryptographic solutions before adversaries gain access to quantum capabilities."
         ]
     },
     quotes: {
         easy: [
             "The only truly secure system is one that is powered off, cast in a block of concrete and sealed in a lead-lined room with armed guards.",
             "Security is always excessive until it fails. Then it wasn't enough.",
             "Privacy is not something that I'm merely entitled to, it's an absolute prerequisite."
         ],
         medium: [
             "If you think technology can solve your security problems, then you don't understand the problems and you don't understand the technology.",
             "Security is not a product, but a process. It's not a technology problem; it's a people and management problem.",
             "The human factor is the weakest link in the security chain, and social engineering is the art of exploiting that weakness."
         ],
         hard: [
             "There are two types of encryption: one that will prevent your sister from reading your diary and one that will prevent your government from reading your diary.",
             "The beautiful thing about defensive programming is the fact that it has no moral component; you're simply trying to make sure your code keeps running regardless of what gets thrown at it.",
             "Encryption works. Properly implemented strong crypto systems are one of the few things that you can rely on. Unfortunately, endpoint security is so terrifically weak that NSA can frequently find ways around it."
         ],
         expert: [
             "Perfect forward secrecy ensures that, if a long-term key is compromised at a later date, it cannot be used to decrypt communications that were recorded in the past, thereby maintaining confidentiality of previous sessions even in the event of catastrophic key exposure.",
             "The principle of least privilege states that a subject should be given only those privileges needed for it to complete its task, thereby reducing the attack surface and minimizing potential damage from accidents, errors, or unauthorized use.",
             "Defense in depth is a security strategy that employs multiple layers of controls throughout an information system, providing redundancy in the event that one security control fails, and creating a comprehensive security posture that addresses different attack vectors simultaneously."
         ]
     },
     code: {
         easy: [
             "function checkPassword(input) {\n  if (input === masterPassword) {\n    return true;\n  }\n  return false;\n}",
             "const encryptData = (data, key) => {\n  let encrypted = [];\n  for (let i = 0; i < data.length; i++) {\n    encrypted.push(data[i] ^ key);\n  }\n  return encrypted;\n}",
             "class Firewall {\n  constructor() {\n    this.rules = [];\n  }\n  addRule(rule) {\n    this.rules.push(rule);\n  }\n  checkAccess(request) {\n    return this.rules.some(rule => rule.matches(request));\n  }\n}"
         ],
         medium: [
             "async function authenticateUser(username, password) {\n  const storedHash = await database.getUserHash(username);\n  if (!storedHash) return { success: false, message: 'User not found' };\n  \n  const passwordMatch = await bcrypt.compare(password, storedHash);\n  if (!passwordMatch) return { success: false, message: 'Invalid password' };\n  \n  return { success: true, token: generateToken(username) };\n}",
             "function detectSQLInjection(input) {\n  const suspiciousPatterns = ['--', ';', '/*', '*/', 'UNION', 'SELECT', 'DROP', 'INSERT', 'DELETE', 'UPDATE'];\n  const normalized = input.toUpperCase();\n  \n  return suspiciousPatterns.some(pattern => normalized.includes(pattern));\n}",
             "class NetworkPacket {\n  constructor(source, destination, data) {\n    this.source = source;\n    this.destination = destination;\n    this.data = data;\n    this.encrypted = false;\n  }\n  \n  encrypt(key) {\n    if (this.encrypted) return;\n    this.data = AES.encrypt(this.data, key);\n    this.encrypted = true;\n  }\n  \n  decrypt(key) {\n    if (!this.encrypted) return;\n    this.data = AES.decrypt(this.data, key);\n    this.encrypted = false;\n  }\n}"
         ],
         hard: [
`import hashlib
import os

def generate_secure_password_hash(password, salt=None):
\"\"\" Generate a secure password hash using PBKDF2 \"\"\"
if salt is None:
 salt = os.urandom(32)  # 32 bytes = 256 bits

# 100,000 iterations of SHA-256
key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)

# Store salt and key together
storage = salt + key
return storage.hex()

def verify_password(stored_hash, provided_password):
\"\"\" Verify a password against the stored hash \"\"\"
import hmac
stored_bytes = bytes.fromhex(stored_hash)
salt = stored_bytes[:32]
stored_key = stored_bytes[32:]
new_key = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
return hmac.compare_digest(new_key, stored_key)
`,

`const protectAgainstXSS = (userInput) => {
// Step 1: Remove dangerous HTML elements
let sanitized = userInput.replace(/<script\\b[^<]*(?:(?!<\\/script>)<[^<]*)*<\\/script>/gi, '');

// Step 2: Escape HTML special characters
sanitized = sanitized
.replace(/&/g, '&amp;')
.replace(/</g, '&lt;')
.replace(/>/g, '&gt;')
.replace(/"/g, '&quot;')
.replace(/'/g, '&#039;');

// Step 3: Validate URLs in attributes
sanitized = sanitized.replace(/href\\s*=\\s*['"](.*?)['"]/, (match, url) => {
if (url.startsWith('javascript:')) {
return 'href="#"';
}
return match;
});

return sanitized;
};`,

`class SecureConnectionManager {
constructor(certificateAuthority) {
this.ca = certificateAuthority;
this.connections = new Map();
this.revocationList = new Set();
}

establishConnection(client, server) {
if (!this.verifyCertificate(client.cert) || !this.verifyCertificate(server.cert)) {
throw new Error('Certificate validation failed');
}

const clientPublic = client.generateKeyPair();
const serverPublic = server.generateKeyPair();

const sessionKey = this.computeSharedSecret(clientPublic, serverPublic);

const connectionId = crypto.randomUUID();
this.connections.set(connectionId, {
client,
server,
established: Date.now(),
sessionKey,
encrypted: true
});

return connectionId;
}

verifyCertificate(cert) {
if (this.revocationList.has(cert.serialNumber)) {
return false;
}

return crypto.verify(
'sha256',
Buffer.from(cert.data),
this.ca.publicKey,
Buffer.from(cert.signature, 'base64')
);
}

computeSharedSecret(publicKeyA, publicKeyB) {
return crypto.createDiffieHellman(publicKeyA).computeSecret(publicKeyB);
}
}`
],
         expert: [
             "package main\n\nimport (\n\t\"crypto/aes\"\n\t\"crypto/cipher\"\n\t\"crypto/rand\"\n\t\"crypto/rsa\"\n\t\"crypto/sha256\"\n\t\"crypto/x509\"\n\t\"encoding/base64\"\n\t\"encoding/pem\"\n\t\"errors\"\n\t\"fmt\"\n\t\"io\"\n\t\"log\"\n)\n\n// HybridEncryption combines RSA and AES for secure data transfer\nfunc HybridEncryption(publicKey *rsa.PublicKey, plaintext []byte) ([]byte, error) {\n\t// Generate a random AES key\n\taesKey := make([]byte, 32) // 256-bit key\n\tif _, err := io.ReadFull(rand.Reader, aesKey); err != nil {\n\t\treturn nil, err\n\t}\n\n\t// Encrypt the plaintext using AES-GCM\n\tblock, err := aes.NewCipher(aesKey)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\n\t// Generate a random nonce\n\tnonce := make([]byte, 12)\n\tif _, err := io.ReadFull(rand.Reader, nonce); err != nil {\n\t\treturn nil, err\n\t}\n\n\taesGCM, err := cipher.NewGCM(block)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\n\t// Encrypt and authenticate the plaintext\n\tciphertext := aesGCM.Seal(nil, nonce, plaintext, nil)\n\n\t// Encrypt the AES key using RSA\n\tencryptedKey, err := rsa.EncryptOAEP(\n\t\tsha256.New(),\n\t\trand.Reader,\n\t\tpublicKey,\n\t\taesKey,\n\t\tnil,\n\t)\n\n\tif err != nil {\n\t\treturn nil, err\n\t}\n\n\t// Package everything together\n\tresult := make([]byte, 0)\n\t\n\t// Format: [keyLength(4)][encryptedKey][nonceLength(4)][nonce][ciphertext]\n\tkeyLen := uint32(len(encryptedKey))\n\tresult = append(result, byte(keyLen>>24), byte(keyLen>>16), byte(keyLen>>8), byte(keyLen))\n\tresult = append(result, encryptedKey...)\n\t\n\tnonceLen := uint32(len(nonce))\n\tresult = append(result, byte(nonceLen>>24), byte(nonceLen>>16), byte(nonceLen>>8), byte(nonceLen))\n\tresult = append(result, nonce...)\n\tresult = append(result, ciphertext...)\n\t\n\treturn result, nil\n}",
             "import java.security.*;\nimport java.security.spec.*;\nimport javax.crypto.*;\nimport javax.crypto.spec.*;\nimport java.util.*;\nimport java.nio.charset.StandardCharsets;\n\npublic class ZeroKnowledgeProofAuthentication {\n    private BigInteger p; // Large prime\n    private BigInteger g; // Generator\n    private BigInteger x; // Secret key\n    private BigInteger y; // Public key\n    \n    // Constructor initializes the parameters\n    public ZeroKnowledgeProofAuthentication(int primeBitLength) {\n        try {\n            // Generate secure parameters\n            SecureRandom random = SecureRandom.getInstanceStrong();\n            \n            // Generate a large prime p\n            p = BigInteger.probablePrime(primeBitLength, random);\n            \n            // Find a generator g of the multiplicative group Z_p*\n            g = findGenerator(p, random);\n            \n            // Generate a random secret key x\n            x = new BigInteger(primeBitLength - 1, random);\n            \n            // Calculate public key y = g^x mod p\n            y = g.modPow(x, p);\n        } catch (NoSuchAlgorithmException e) {\n            throw new RuntimeException(\"Secure random algorithm not available\", e);\n        }\n    }\n    \n    // The prover wants to prove they know x without revealing it\n    public Map<String, BigInteger> generateProof() {\n        try {\n            SecureRandom random = SecureRandom.getInstanceStrong();\n            \n            // Prover chooses a random value k\n            BigInteger k = new BigInteger(p.bitLength() - 1, random);\n            \n            // Compute r = g^k mod p\n            BigInteger r = g.modPow(k, p);\n            \n            // Generate challenge (in a real scenario, this would come from the verifier)\n            byte[] challengeBytes = r.toByteArray();\n            MessageDigest digest = MessageDigest.getInstance(\"SHA-256\");\n            byte[] hashBytes = digest.digest(challengeBytes);\n            BigInteger c = new BigInteger(1, hashBytes).mod(p);\n            \n            // Compute s = (k - c*x) mod (p-1)\n            BigInteger s = k.subtract(c.multiply(x)).mod(p.subtract(BigInteger.ONE));\n            \n            // Return the proof components\n            Map<String, BigInteger> proof = new HashMap<>();\n            proof.put(\"r\", r);\n            proof.put(\"s\", s);\n            proof.put(\"c\", c);\n            return proof;\n        } catch (Exception e) {\n            throw new RuntimeException(\"Failed to generate proof\", e);\n        }\n    }\n    \n    // The verifier checks the proof\n    public boolean verifyProof(Map<String, BigInteger> proof) {\n        BigInteger r = proof.get(\"r\");\n        BigInteger s = proof.get(\"s\");\n        BigInteger c = proof.get(\"c\");\n        \n        // Verify that g^s * y^c = r mod p\n        BigInteger leftSide = g.modPow(s, p).multiply(y.modPow(c, p)).mod(p);\n        return leftSide.equals(r);\n    }\n    \n    // Helper method to find a generator of Z_p*\n    private BigInteger findGenerator(BigInteger p, SecureRandom random) {\n        BigInteger pMinusOne = p.subtract(BigInteger.ONE);\n        // For simplicity, we're looking for random elements until we find a generator\n        // In practice, more efficient algorithms would be used\n        \n        while (true) {\n            BigInteger candidate = new BigInteger(p.bitLength() - 1, random);\n            if (candidate.compareTo(BigInteger.ONE) > 0 && \n                candidate.compareTo(p) < 0 && \n                candidate.modPow(pMinusOne, p).equals(BigInteger.ONE)) {\n                return candidate;\n            }\n        }\n    }\n}",
             "# This is a machine learning-based intrusion detection system implementation\n\nimport numpy as np\nimport pandas as pd\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import classification_report, confusion_matrix\n\nclass NetworkIntrusionDetectionSystem:\n    def __init__(self):\n        self.model = RandomForestClassifier(n_estimators=100, random_state=42)\n        self.scaler = StandardScaler()\n        self.feature_columns = None\n        self.trained = False\n    \n    def preprocess_data(self, data):\n        \"\"\"Preprocess network traffic data for analysis\"\"\"\n        # Convert categorical features to numerical\n        categorical_cols = data.select_dtypes(include=['object']).columns\n        \n        for col in categorical_cols:\n            data[col] = pd.Categorical(data[col]).codes\n        \n        # Handle missing values\n        data = data.fillna(0)\n        \n        return data\n    \n    def extract_features(self, data):\n        \"\"\"Extract relevant features from network traffic\"\"\"\n        # Remove non-predictive columns like timestamps or IDs\n        if 'timestamp' in data.columns:\n            data = data.drop('timestamp', axis=1)\n        if 'packet_id' in data.columns:\n            data = data.drop('packet_id', axis=1)\n        \n        # Store feature columns for prediction\n        self.feature_columns = data.columns.drop('is_attack')\n        \n        # Split features and target\n        X = data[self.feature_columns]\n        y = data['is_attack']\n        \n        # Scale numerical features\n        X = self.scaler.fit_transform(X)\n        \n        return X, y\n    \n    def train(self, training_data):\n        \"\"\"Train the machine learning model on network traffic data\"\"\"\n        # Preprocess the data\n        processed_data = self.preprocess_data(training_data)\n        \n        # Extract features\n        X, y = self.extract_features(processed_data)\n        \n        # Split into train and validation sets\n        X_train, X_val, y_train, y_val = train_test_split(\n            X, y, test_size=0.2, random_state=42)\n        \n        # Train the model\n        self.model.fit(X_train, y_train)\n        \n        # Evaluate on validation set\n        y_pred = self.model.predict(X_val)\n        \n        print(\"Intrusion Detection System Training Results:\")\n        print(confusion_matrix(y_val, y_pred))\n        print(classification_report(y_val, y_pred))\n        \n        self.trained = True\n        return self\n    \n    def detect_intrusions(self, network_traffic):\n        \"\"\"Detect potential intrusions in network traffic\"\"\"\n        if not self.trained:\n            raise ValueError(\"Model not trained yet! Call train() first.\")\n        \n        # Preprocess new data\n        processed_data = self.preprocess_data(network_traffic)\n        \n        # Ensure we have all needed columns\n        missing_cols = set(self.feature_columns) - set(processed_data.columns)\n        for col in missing_cols:\n            processed_data[col] = 0\n        \n        # Use only relevant features in the correct order\n        X_new = processed_data[self.feature_columns]\n        \n        # Scale features\n        X_new = self.scaler.transform(X_new)\n        \n        # Make predictions\n        predictions = self.model.predict(X_new)\n        probabilities = self.model.predict_proba(X_new)[:, 1]\n        \n        # Return results\n        results = pd.DataFrame({\n            'is_intrusion': predictions,\n            'confidence': probabilities\n        })\n        \n        return results\n    \n    def update_model(self, new_training_data):\n        \"\"\"Update the model with new training data\"\"\"\n        # Combine with existing model knowledge (transfer learning)\n        # This is a simplified implementation\n        self.train(new_training_data)\n        return self"
         ]
     }
 };
 
 // Typing tips
 const typingTips = [
     "Place your fingers on the home row keys: A, S, D, F for your left hand and J, K, L, ; for your right hand.",
     "Don't look at the keyboard while typing. Try to memorize the key positions.",
     "Start slowly and focus on accuracy rather than speed. Speed will come with practice.",
     "Take short breaks if you feel your hands getting tired. Proper ergonomics is important.",
     "Practice regularly, even just for 10-15 minutes a day, to build muscle memory.",
     "Use your pinky fingers for keys like Shift, Enter, and Backspace.",
     "Maintain good posture with your back straight and wrists slightly elevated.",
     "Challenge yourself with progressively harder texts as you improve.",
     "Focus on problem keys or combinations that slow you down.",
     "Remember that everyone has a different typing style. Find what works best for you."
 ];
 
 // Feedback messages
 const feedbackMessages = {
     great: ["Perfect!", "Excellent!", "Impressive!", "Outstanding!", "Superb!"],
     good: ["Great job!", "Well done!", "Nice typing!", "Keep it up!", "You're on fire!"],
     average: ["Good pace.", "Doing well.", "Keep going.", "Stay focused.", "You can do it!"],
     poor: ["Slow down.", "Focus on accuracy.", "Careful now.", "Take your time.", "Watch for errors."],
     combo: ["Combo x", "Streak x", "Chain x", "Sequence x", "Series x"]
 };
 
 // Mission descriptions
 const missionDescriptions = [
     "// MISSION: Breach the firewall by typing the exact code sequence",
     "// MISSION: Decrypt the security protocols by entering the correct text",
     "// MISSION: Bypass the authentication system with precise keystrokes",
     "// MISSION: Infiltrate the mainframe with accurate typing",
     "// MISSION: Override security measures with flawless input"
 ];
 
 // Leaderboard data (mock data)
 const leaderboardData = [
     { name: "CyberNinja", wpm: 120, accuracy: 98, level: 10 },
     { name: "HackMaster", wpm: 115, accuracy: 97, level: 9 },
     { name: "CodePhantom", wpm: 108, accuracy: 99, level: 8 },
     { name: "KeyboardWizard", wpm: 105, accuracy: 96, level: 8 },
     { name: "DigitalShadow", wpm: 102, accuracy: 95, level: 7 },
     { name: "NeuralHacker", wpm: 98, accuracy: 94, level: 7 },
     { name: "QuantumTyper", wpm: 95, accuracy: 96, level: 6 },
     { name: "CipherBreaker", wpm: 90, accuracy: 97, level: 6 },
     { name: "SyntaxSpecter", wpm: 88, accuracy: 93, level: 5 },
     { name: "BitCrusher", wpm: 85, accuracy: 92, level: 5 }
 ];
 
 // Settings
 const settings = {
     difficulty: "medium",
     mode: "quotes",
     soundEffects: true,
     visualEffects: true
 };
 
 // Initialize game
 function initializeGame() {

        // Show username modal first
        document.getElementById('username-modal').classList.add('active');
    
        // Set up event listeners
        document.getElementById('start-game-btn').addEventListener('click', function() {
            username = document.getElementById('username-input').value.trim();
            if (!username) username = 'Anonymous';
            document.getElementById('username-modal').classList.remove('active');
        });
        
     // Set up event listeners
     textInput.addEventListener('input', handleInput);
     startBtn.addEventListener('click', startGame);
     restartBtn.addEventListener('click', restartGame);
     settingsBtn.addEventListener('click', openSettings);
     leaderboardBtn.addEventListener('click', openLeaderboard);
     closeSettings.addEventListener('click', closeModal);
     closeLeaderboard.addEventListener('click', closeModal);
     saveSettings.addEventListener('click', saveGameSettings);
     nextLevelBtn.addEventListener('click', nextLevel);
     retryBtn.addEventListener('click', retryLevel);
     viewStatsBtn.addEventListener('click', viewStats);
     
     // Initial setup
     updateLeaderboard();
     createParticles();
 }
 
 // Start the game
 function startGame() {
     if (gameActive) return;
     
     gameActive = true;
     startBtn.style.display = 'none';
     restartBtn.style.display = 'inline-block';
     gameOverScreen.style.display = 'none';
     
     // Reset variables
     errors = 0;
     combo = 0;
     currentIndex = 0;
     typedText = '';
     
     // Get text based on settings
     currentText = getRandomText();
     
     // Display text
     displayText();
     
     // Set random mission prompt
     const randomMission = missionDescriptions[Math.floor(Math.random() * missionDescriptions.length)];
     missionPrompt.textContent = randomMission;
     
     // Focus on input
     textInput.value = '';
     textInput.focus();
     
     // Record start time
     startTime = Date.now();
     
     // Update display
     updateStats();
     updateCombo();
     
     // Start periodic updates
     requestAnimationFrame(updateGame);
 }
 
 // Get random text based on current settings
 function getRandomText() {
     const mode = settings.mode;
     const difficulty = settings.difficulty;
     
     const textsInCategory = textContent[mode][difficulty];
     return textsInCategory[Math.floor(Math.random() * textsInCategory.length)];
 }
 
 // Display text in the game area
// Updated displayText function in script.js
function displayText() {
    textDisplay.innerHTML = '';

    // Create a <pre> element to handle multiline code blocks with indentation
    const pre = document.createElement('pre');
    pre.style.margin = '0';
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.lineHeight = '1.5';

    for (let i = 0; i < currentText.length; i++) {
        const span = document.createElement('span');
        span.textContent = currentText[i];
        pre.appendChild(span);
    }

    textDisplay.appendChild(pre);

    // Mark first character as current
    if (pre.children.length > 0) {
        pre.children[0].classList.add('current');
    }

    // FIX: Force scroll to top of the text container (not just the page)
    const textContainer = textDisplay.closest('.text-container');
    if (textContainer) {
        textContainer.scrollTo({ top: 0, behavior: 'instant' });
    }

    // FIX: Prevent double scrolling — remove overflow from textDisplay if nested scrollbars exist
    textDisplay.style.overflow = 'visible';
}


 
 // Handle input
 function handleInput(e) {
     if (!gameActive) return;
     
     const inputChar = e.data;
     
     if (inputChar === null) {
         // Handle backspace
         if (currentIndex > 0 && textInput.value.length < currentIndex) {
             currentIndex--;
             updateTextDisplay();
         }
         return;
     }
     
     // Check if the character is correct
     const currentChar = currentText[currentIndex];
     const isCorrect = inputChar === currentChar;

     
     // Add screen shake for errors
    if (!isCorrect) {
        document.querySelector('.terminal').classList.add('shake');
        setTimeout(() => {
            document.querySelector('.terminal').classList.remove('shake');
        }, 500);
    }
    
    // Add matrix effect during combos
    if (combo % 20 === 0 && combo > 0) {
        createMatrixEffect();
    }
     
     // Update display
     const spans = textDisplay.querySelectorAll('span');
     spans[currentIndex].classList.remove('current');
     
     if (isCorrect) {
         spans[currentIndex].classList.add('correct');
         combo++;
         
         // Show particle effect on correct typing
         if (settings.visualEffects) {
             createTypingParticle(spans[currentIndex], true);
         }
         
         // Play sound effect
         if (settings.soundEffects) {
             playSound('correct');
         }
         
         // Show feedback occasionally
         if (combo % 10 === 0) {
             showFeedback('good');
         }
     } else {
         spans[currentIndex].classList.add('incorrect');
         errors++;
         combo = 0;
         
         // Show particle effect on incorrect typing
         if (settings.visualEffects) {
             createTypingParticle(spans[currentIndex], false);
         }
         
         // Play sound effect
         if (settings.soundEffects) {
             playSound('incorrect');
         }
         
         // Show feedback for errors
         showFeedback('poor');
     }
     
     // Move to next character
     currentIndex++;
     typedText += inputChar;
     
     // Check if we've reached the end of the text
     if (currentIndex >= currentText.length) {
         gameComplete();
         return;
     }
     
     // Mark next character as current
     spans[currentIndex].classList.add('current');
     
     // Update stats
     updateStats();
     updateCombo();
 }

 function createMatrixEffect() {
    const matrix = document.createElement('div');
    matrix.className = 'matrix-effect';
    document.body.appendChild(matrix);
    
    setTimeout(() => {
        matrix.remove();
    }, 2000);
}
 
 // Hacking animation variables
const hackMessages = [
    "ACCESS GRANTED",
    "FIREWALL BREACHED",
    "ENCRYPTION BYPASSED",
    "SYSTEM INFILTRATED",
    "SECURITY OVERRIDE",
    "ROOT ACCESS OBTAINED"
];

const codeSnippets = [
    "[+] Establishing secure connection...\n[+] Bypassing firewall...\n[+] Decrypting security protocols...\n[+] Gaining elevated privileges...\n[+] Access granted to mainframe...",
    "Initializing exploit sequence...\nInjecting payload...\nOverriding security checks...\nCreating backdoor entry...\nSystem compromised successfully...",
    "Scanning vulnerabilities...\nExploiting CVE-2023-XXXXX...\nUploading rootkit...\nCovering tracks...\nPersistent access established...",
    "Running privilege escalation...\nDisabling security monitoring...\nDumping credentials...\nPivoting to internal network...\nMission accomplished..."
];
 // Update text display
 function updateTextDisplay() {
     const spans = textDisplay.querySelectorAll('span');
     
     // Reset all spans
     for (let i = 0; i < spans.length; i++) {
         spans[i].classList.remove('current', 'correct', 'incorrect');
     }
     
     // Mark characters up to currentIndex
     for (let i = 0; i < currentIndex; i++) {
         if (i < typedText.length) {
             if (typedText[i] === currentText[i]) {
                 spans[i].classList.add('correct');
             } else {
                 spans[i].classList.add('incorrect');
             }
         }
     }
     
     // Mark current character
     if (currentIndex < spans.length) {
         spans[currentIndex].classList.add('current');
     }
 }
 
 // Update stats display
 function updateStats() {
     const comboMultiplier = 1 + (combo * 0.02);
     const elapsedTime = Math.max(1, (Date.now() - startTime) / 1000);
     const words = currentIndex / 5; // Assuming average word length of 5 characters
     wpm = Math.round(words / (elapsedTime / 60)) * comboMultiplier;
     
     const totalCharactersTyped = typedText.length;
     accuracy = totalCharactersTyped > 0 
         ? Math.round(((totalCharactersTyped - errors) / totalCharactersTyped) * 100) 
         : 100;
     
     wpmDisplay.textContent = wpm;
     accuracyDisplay.textContent = accuracy + '%';
     errorsDisplay.textContent = errors;
     levelDisplay.textContent = gameLevel;
 }
 
 // Update combo display
 function updateCombo() {
     const comboPercentage = Math.min(100, combo * 2);
     comboFill.style.height = comboPercentage + '%';
     comboValue.textContent = 'x' + combo;
 }
 
 // Show feedback message
 function showFeedback(type) {
     const messages = feedbackMessages[type];
     const message = messages[Math.floor(Math.random() * messages.length)];
     
     feedbackDisplay.textContent = type === 'combo' ? message + combo : message;
     feedbackDisplay.classList.add('active');
     
     setTimeout(() => {
         feedbackDisplay.classList.remove('active');
     }, 1500);
 }
 
 // Game complete
 function gameComplete() {
    gameActive = false;
    
    // Show mission passed animation
    showMissionPassed();
    
    // Automatically progress to next level after delay
    setTimeout(() => {
        gameLevel++;
        startGame();
    }, 4000); // 3 second delay before next level
    
    // Calculate final stats
    const elapsedTime = Math.round((Date.now() - startTime) / 1000);
    
    // Store progress
    progressHistory.push({
        level: gameLevel,
        wpm,
        accuracy,
        time: elapsedTime
    });
    
    // Show hacking success animation
    showHackingAnimation();
}

function showMissionPassed() {
    const missionPassed = document.createElement('div');
    missionPassed.className = 'mission-passed';
    missionPassed.textContent = 'MISSION PASSED';
    document.querySelector('.terminal').appendChild(missionPassed);
    
    setTimeout(() => {
        missionPassed.remove();
    }, 2500);
}

// Show hacking success animation
function showHackingAnimation() {
    const hackSuccess = document.getElementById('hack-success');
    const hackMessage = document.getElementById('hack-message');
    const hackProgress = document.getElementById('hack-progress');
    const codeStream = document.getElementById('code-stream');
    
    // Set random hacking message
    hackMessage.textContent = hackMessages[Math.floor(Math.random() * hackMessages.length)];
    
    // Set random code stream
    codeStream.textContent = codeSnippets[Math.floor(Math.random() * codeSnippets.length)];
    
    // Show the animation
    hackSuccess.classList.add('active');
    
    // Animate progress bar
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 2;
        hackProgress.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
            
            // Hide animation and show results after delay
            setTimeout(() => {
                hackSuccess.classList.remove('active');
                showGameResults();
            }, 1000);
        }
    }, 50);
}

function showGameResults() {
    // Update result display
    resultWpm.textContent = wpm;
    resultAccuracy.textContent = accuracy + '%';
    resultTime.textContent = Math.round((Date.now() - startTime) / 1000) + 's';
    resultLevel.textContent = gameLevel;
    
    // Show game over screen with "Mission Passed" message
    gameOverScreen.style.display = 'flex';
    gameOverScreen.querySelector('h2').textContent = 'MISSION PASSED';
    
    // Show a random tip
    tipBox.textContent = 'Pro Tip: ' + typingTips[Math.floor(Math.random() * typingTips.length)];
    
    // Create progress chart
    createProgressChart();
    
    // Add to leaderboard with username
    if (wpm > 50 && accuracy > 90) {
        leaderboardData.push({
            name: username,
            wpm,
            accuracy,
            level: gameLevel
        });
        
        // Sort leaderboard by WPM
        leaderboardData.sort((a, b) => b.wpm - a.wpm);
        
        // Keep only top 10
        if (leaderboardData.length > 10) {
            leaderboardData.pop();
        }
        
        // Update leaderboard
        updateLeaderboard();
    }
    
    textInput.blur();
}
 
 // Create progress chart
 function createProgressChart() {
     progressChart.innerHTML = '';
     
     // Only show the last 5 entries if there are more
     const chartData = progressHistory.slice(-5);
     
     const maxWpm = Math.max(...chartData.map(entry => entry.wpm), 1);
     const chartWidth = progressChart.clientWidth;
     const barWidth = Math.min(40, chartWidth / chartData.length - 10);
     
     chartData.forEach((entry, index) => {
         const barHeight = (entry.wpm / maxWpm) * 170;
         
         const bar = document.createElement('div');
         bar.className = 'chart-bar';
         bar.style.height = '0';
         bar.style.left = (chartWidth - barWidth * chartData.length) / 2 + index * barWidth + index * 10 + 'px';
         bar.style.width = barWidth + 'px';
         
         const label = document.createElement('div');
         label.className = 'chart-label';
         label.textContent = 'L' + entry.level;
         label.style.left = (chartWidth - barWidth * chartData.length) / 2 + index * barWidth + index * 10 + 'px';
         label.style.width = barWidth + 'px';
         
         progressChart.appendChild(bar);
         progressChart.appendChild(label);
         
         // Animate the bar
         setTimeout(() => {
             bar.style.height = barHeight + 'px';
         }, 100 * index);
     });
 }
 
 // Next level
 function nextLevel() {
     gameLevel++;
     startGame();
 }
 
 // Retry level
 function retryLevel() {
     startGame();
 }
 
 // View stats (placeholder function)
 function viewStats() {
     // This would show a more detailed stats screen
     // For now, just reuse the leaderboard modal
     openLeaderboard();
 }
 
 // Restart game
 function restartGame() {
     gameActive = false;
     gameLevel = 1;
     progressHistory = [];
     startGame();
 }
 
 // Open settings modal
 function openSettings() {
     settingsModal.classList.add('active');
     
     // Set current settings
     difficultySelect.value = settings.difficulty;
     modeSelect.value = settings.mode;
     soundEffectsCheck.checked = settings.soundEffects;
     visualEffectsCheck.checked = settings.visualEffects;
 }
 
 // Open leaderboard modal
 function openLeaderboard() {
     leaderboardModal.classList.add('active');
     updateLeaderboard();
 }
 
 // Close modal
 function closeModal() {
     settingsModal.classList.remove('active');
     leaderboardModal.classList.remove('active');
 }
 
 // Save game settings
 function saveGameSettings() {
     settings.difficulty = difficultySelect.value;
     settings.mode = modeSelect.value;
     settings.soundEffects = soundEffectsCheck.checked;
     settings.visualEffects = visualEffectsCheck.checked;
     
     closeModal();
 }
 
 // Update leaderboard
 function updateLeaderboard() {
     leaderboardBody.innerHTML = '';
     
     leaderboardData.forEach((entry, index) => {
         const row = document.createElement('tr');
         
         const rankCell = document.createElement('td');
         rankCell.textContent = index + 1;
         
         const nameCell = document.createElement('td');
         nameCell.textContent = entry.name;
         
         const wpmCell = document.createElement('td');
         wpmCell.textContent = entry.wpm;
         
         const accuracyCell = document.createElement('td');
         accuracyCell.textContent = entry.accuracy + '%';
         
         const levelCell = document.createElement('td');
         levelCell.textContent = entry.level;
         
         row.appendChild(rankCell);
         row.appendChild(nameCell);
         row.appendChild(wpmCell);
         row.appendChild(accuracyCell);
         row.appendChild(levelCell);
         
         leaderboardBody.appendChild(row);
     });
 }
 
 // Create background particles
 function createParticles() {
     const container = document.querySelector('.container');
     
     for (let i = 0; i < 30; i++) {
         const particle = document.createElement('div');
         particle.className = 'particle';
         
         // Random properties
         const size = Math.random() * 5 + 2;
         const posX = Math.random() * 100;
         const posY = Math.random() * 100;
         const animationDuration = Math.random() * 30 + 10;
         const animationDelay = Math.random() * 5;
         
         // Apply styles
         particle.style.width = size + 'px';
         particle.style.height = size + 'px';
         particle.style.left = posX + '%';
         particle.style.top = posY + '%';
         particle.style.animation = `float ${animationDuration}s infinite ease-in-out ${animationDelay}s`;
         
         container.appendChild(particle);
     }
 }
 
 // Create typing particle effect
 function createTypingParticle(element, isCorrect) {
     const rect = element.getBoundingClientRect();
     const x = rect.left + rect.width / 2;
     const y = rect.top + rect.height / 2;
     
     for (let i = 0; i < 5; i++) {
         const particle = document.createElement('div');
         particle.className = 'particle';
         
         // Set color based on correctness
         if (isCorrect) {
             particle.style.background = 'var(--primary)';
         } else {
             particle.style.background = 'var(--error)';
         }
         
         // Random size
         const size = Math.random() * 3 + 1;
         particle.style.width = size + 'px';
         particle.style.height = size + 'px';
         
         // Position at the character
         particle.style.left = x + 'px';
         particle.style.top = y + 'px';
         
         // Random direction
         const angle = Math.random() * Math.PI * 2;
         const speed = Math.random() * 30 + 10;
         const vx = Math.cos(angle) * speed;
         const vy = Math.sin(angle) * speed;
         
         // Append to body
         document.body.appendChild(particle);
         
         // Animate
         setTimeout(() => {
             particle.style.transition = 'all 0.5s ease-out';
             particle.style.left = (x + vx) + 'px';
             particle.style.top = (y + vy) + 'px';
             particle.style.opacity = '0';
         }, 10);
         
         // Remove after animation
         setTimeout(() => {
             particle.remove();
         }, 500);
     }
 }
 
 // Play sound effect (placeholder function)
 function playSound(type) {
     // In a real implementation, this would play actual sounds
     // For now, we'll use the Web Audio API to generate simple sounds
     
     if (!settings.soundEffects) return;
     
     const audioContext = new (window.AudioContext || window.webkitAudioContext)();
     const oscillator = audioContext.createOscillator();
     const gainNode = audioContext.createGain();
     
     oscillator.connect(gainNode);
     gainNode.connect(audioContext.destination);
     
     if (type === 'correct') {
         oscillator.type = 'sine';
         oscillator.frequency.value = 660;
         gainNode.gain.value = 0.1;
         
         oscillator.start();
         oscillator.stop(audioContext.currentTime + 0.08);
     } else if (type === 'incorrect') {
         oscillator.type = 'sawtooth';
         oscillator.frequency.value = 220;
         gainNode.gain.value = 0.1;
         
         oscillator.start();
         oscillator.stop(audioContext.currentTime + 0.15);
     }
 }
 
 // Update game state
 function updateGame() {
     if (gameActive) {
         updateStats();
         requestAnimationFrame(updateGame);
     }
 }
 
 // Initialize the game when the page loads
 window.addEventListener('load', initializeGame);