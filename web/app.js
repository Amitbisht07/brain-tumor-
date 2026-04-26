
const API = "http://127.0.0.1:5000"; // Flask backend

const setToken = (t) => localStorage.setItem("jwt", t);
const getToken = () => localStorage.getItem("jwt");
const clearToken = () => localStorage.removeItem("jwt");

const setLastEmail = (email) => localStorage.setItem("last_email", email);
const getLastEmail = () => localStorage.getItem("last_email") || "";
const clearLastEmail = () => localStorage.removeItem("last_email");

async function doLogin(e) {
  e.preventDefault();

  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value.trim();
  const msg = document.getElementById("authMsg");

  msg.textContent = "Verifying credentials...";
  msg.style.color = "#333";

  try {
    const res = await fetch(`${API}/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });

    const data = await res.json();
    console.log("Login response:", data);

    if (!data.success) {
      msg.textContent = data.message || "Login failed";
      msg.style.color = "red";
      return false;
    }

    
    setToken(data.token);
    setLastEmail(email);

    msg.textContent = "✅ Login successful! Redirecting...";
    msg.style.color = "green";

    setTimeout(() => (window.location.href = "index.html"), 1000);
  } catch (err) {
    console.error("Login error:", err);
    msg.textContent = "Server error. Please try again.";
    msg.style.color = "red";
  }

  return false;
}


async function doSignup(e) {
  e.preventDefault();

  const name = document.getElementById("name").value.trim();
  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value.trim();
  const msg = document.getElementById("authMsg");

  msg.textContent = "Creating account...";
  msg.style.color = "#333";

  try {
    const res = await fetch(`${API}/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    });

    const data = await res.json();
    console.log("Signup response:", data);

    if (!data.success) {
      msg.textContent = data.message || "Signup failed";
      msg.style.color = "red";
      return false;
    }

    
    setLastEmail(email);

    msg.textContent = "✅ Account created successfully! Redirecting to login...";
    msg.style.color = "green";

    setTimeout(() => (window.location.href = "login.html"), 1200);
  } catch (err) {
    console.error("Signup error:", err);
    msg.textContent = "Server error.";
    msg.style.color = "red";
  }

  return false;
}


// ✅ LOGOUT
// ==========================================
function logout() {
  clearToken();
  window.location.href = "login.html";
}


document.addEventListener("DOMContentLoaded", () => {
  const page = document.body.dataset.page;

 
  if (page === "login") {
    const lastEmail = getLastEmail();
    if (lastEmail) {
      const emailInput = document.getElementById("email");
      emailInput.value = lastEmail;
      const msg = document.getElementById("authMsg");
      msg.textContent = `Welcome back, ${lastEmail}`;
      msg.style.color = "#3478ff";
    }

    if (getToken()) {
      window.location.href = "index.html";
    }

    document.getElementById("loginForm").addEventListener("submit", doLogin);
  }

  // Signup Page
  if (page === "signup") {
    document.getElementById("signupForm").addEventListener("submit", doSignup);
  }

  // Dashboard Page
  if (page === "dashboard") {
    if (!getToken()) {
      alert("Please login first.");
      window.location.href = "login.html";
    }

    const lo = document.getElementById("logoutBtn");
    if (lo) lo.addEventListener("click", logout);
  }
});

async function submitForm() {
  const token = getToken();
  if (!token) {
    alert("Please login first.");
    window.location.href = "login.html";
    return;
  }

  const name = document.getElementById("name").value.trim();
  const age = document.getElementById("age").value.trim();
  const gender = document.getElementById("gender").value;
  const notes = document.getElementById("notes").value.trim();
  const file = document.getElementById("imageUpload").files[0];
  const result = document.getElementById("result");
  const loading = document.getElementById("loading");

  if (!file) {
    alert("Please upload an MRI image.");
    return;
  }

  const fd = new FormData();
  fd.append("name", name);
  fd.append("age", age);
  fd.append("gender", gender);
  fd.append("notes", notes);
  fd.append("file", file);

  result.textContent = "";
  loading.classList.remove("hidden");

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { Authorization: token },
      body: fd,
    });

    const data = await res.json();
    console.log("Prediction response:", data);

    loading.classList.add("hidden");

    if (data.error) {
      result.textContent = "❌ " + data.error;
      result.className = "result-box low";
    } else {
      const conf = data.confidence || 0;
      let cls = conf >= 90 ? "high" : conf >= 60 ? "mid" : "low";
      result.className = `result-box ${cls}`;
      result.innerHTML = `
        🧠 <b>${data.prediction.toUpperCase()}</b><br>
        Confidence: ${conf.toFixed(2)}%
      `;
    }
  } catch (err) {
    console.error("Prediction error:", err);
    loading.classList.add("hidden");
    result.textContent = "Server error during prediction.";
    result.className = "result-box low";
  }
}
