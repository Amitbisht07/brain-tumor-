document.addEventListener("DOMContentLoaded", async () => {
  const token = localStorage.getItem("jwt");
  const body = document.getElementById("historyBody");

  if (!token) {
    alert("Please login first.");
    window.location.href = "login.html";
    return;
  }

  try {
    const res = await fetch("http://127.0.0.1:5000/history", {
      method: "GET",
      headers: { Authorization: token },
    });

    const data = await res.json();
    console.log("History data:", data);

    body.innerHTML = "";

    if (!data || data.length === 0) {
      body.innerHTML = `<tr><td colspan="6" class="mute">No records found yet.</td></tr>`;
      return;
    }

    data.forEach((r) => {
      const conf = +r.confidence || 0;
      const cls = conf >= 90 ? "high" : conf >= 60 ? "mid" : "low";

      const row = `
        <tr>
          <td>${r.name || "-"}</td>
          <td>${r.age || "-"}</td>
          <td>${r.gender || "-"}</td>
          <td><b>${r.prediction || "-"}</b></td>
          <td><span class="pill ${cls}">${conf.toFixed(2)}%</span></td>
          <td>${new Date(r.timestamp).toLocaleString()}</td>
        </tr>`;
      body.innerHTML += row;
    });
  } catch (err) {
    console.error("Error fetching history:", err);
    body.innerHTML = `<tr><td colspan="6" class="low">Error loading history.</td></tr>`;
  }
});
