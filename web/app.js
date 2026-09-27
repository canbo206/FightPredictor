"use strict";

const $ = (id) => document.getElementById(id);
const form = $("matchup-form");
let busy = false;
let lastResult = null;

async function request(url, options = {}) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(20000) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed. Please try again.");
  return data;
}

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function format(value) {
  return value === null ? "N/A" : typeof value === "string" ? value : value.toFixed(2);
}

function render(result) {
  lastResult = result;
  for (const [corner, key] of [["red", "fighter1"], ["blue", "fighter2"]]) {
    $(`${corner}-name`).textContent = result[key];
    $(`${corner}-column`).textContent = result[key];
    const probability = result.probabilities[key] * 100;
    const percent = document.createElement("span");
    percent.className = "percent";
    percent.textContent = "%";
    $(`${corner}-probability`).replaceChildren(document.createTextNode(probability.toFixed(1)), percent);
    $(`${corner}-track`).style.width = `${probability}%`;
  }
  $("bout-label").textContent = `${result.rounds}-ROUND BOUT`;
  $("tracked-fights").textContent = `${result.tracked_fights.fighter1} / ${result.tracked_fights.fighter2}`;
  $("data-note").textContent = `As of ${result.as_of} · Probabilities adjusted on a separate calibration period. No market odds included.`;
  $("data-warning").textContent = result.warnings.join(" ");
  $("data-warning").hidden = result.warnings.length === 0;
  const difference = result.probabilities.fighter1 - result.probabilities.fighter2;
  $("favored").textContent = Math.abs(difference) < 0.00001 ? "An even matchup" : `${difference > 0 ? result.fighter1 : result.fighter2} is favored`;

  $("methods").replaceChildren(...["KO/TKO", "Submission", "Decision"].map((method) => {
    const probability = (result.method_probabilities[method] || 0) * 100;
    const card = document.createElement("article");
    card.className = `method-card${result.predicted_method === method ? " likely" : ""}`;
    const top = document.createElement("div");
    top.className = "method-top";
    const name = document.createElement("span");
    name.className = "method-name";
    name.textContent = method;
    top.append(name);
    if (result.predicted_method === method) {
      const badge = document.createElement("span");
      badge.className = "likely-label";
      badge.textContent = "MOST LIKELY";
      top.append(badge);
    }
    const value = document.createElement("p");
    value.className = "method-value";
    value.textContent = `${probability.toFixed(1)}%`;
    const track = document.createElement("div");
    track.className = "method-track";
    track.setAttribute("aria-hidden", "true");
    const fill = document.createElement("div");
    fill.className = "method-fill";
    fill.style.width = `${probability}%`;
    track.append(fill);
    card.append(top, value, track);
    return card;
  }));

  $("stats").replaceChildren(...result.stats.map((stat) => {
    const row = document.createElement("tr");
    let edge = stat.edge === "even" ? "Neutral" : "—";
    if (stat.edge && stat.edge !== "even") edge = result[stat.edge];
    for (const text of [stat.label, format(stat.fighter1), format(stat.fighter2), edge]) {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.append(cell);
    }
    row.lastChild.className = stat.edge === "fighter1" ? "red-text" : stat.edge === "fighter2" ? "blue-text" : "even";
    row.lastChild.title = `Contribution to fighter 1's log-odds: ${stat.contribution.toFixed(3)}. Positive favors fighter 1; negative favors fighter 2.`;
    return row;
  }));
  $("empty").hidden = true;
  $("results").hidden = false;
  $("activity").textContent = `Analysis ready: ${result.fighter1} versus ${result.fighter2}. ${$("favored").textContent}.`;
}

async function analyze(event) {
  event?.preventDefault();
  if (busy || !form.reportValidity()) return;
  const fighter1 = $("fighter1").value.trim();
  const fighter2 = $("fighter2").value.trim();
  if (!fighter1 || !fighter2) return showError("Enter two fighter names.");
  if (fighter1.toLowerCase() === fighter2.toLowerCase()) return showError("Choose two different fighters.");
  busy = true;
  showError("");
  $("results").hidden = true;
  $("empty").hidden = true;
  $("predict-button").disabled = true;
  $("predict-button").textContent = "Analyzing…";
  form.setAttribute("aria-busy", "true");
  $("activity").textContent = "Calculating matchup probabilities.";
  // Keep the submitted matchup and controls in sync while a request is running.
  const inputs = [...form.querySelectorAll("input")];
  const rounds = Number(new FormData(form).get("rounds"));
  inputs.forEach((input) => { input.disabled = true; });
  try {
    render(await request("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fighter1, fighter2, rounds }),
    }));
  } catch (error) {
    showError(error.name === "TimeoutError" ? "The prediction took too long. Check the local server and retry." : error.message);
    $("empty").hidden = false;
    $("activity").textContent = "Prediction failed.";
  } finally {
    busy = false;
    inputs.forEach((input) => { input.disabled = false; });
    $("predict-button").disabled = false;
    $("predict-button").textContent = "Analyze matchup ↗";
    form.setAttribute("aria-busy", "false");
  }
}

async function loadFighters() {
  $("retry").hidden = true;
  $("connection").textContent = "Loading fighters from your database…";
  try {
    const { fighters } = await request("/api/fighters");
    $("fighters").replaceChildren(...fighters.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      return option;
    }));
    $("connection").textContent = fighters.length ? `${fighters.length.toLocaleString()} fighters available · Predictions use your saved models` : "No fighter statistics yet. Run the scraper to populate your database, then retry.";
    $("retry").hidden = fighters.length > 0;
    // Open with the same example matchup as the terminal, when both fighters exist.
    if (!$("fighter1").value && !$("fighter2").value && fighters.includes("Islam Makhachev") && fighters.includes("Dustin Poirier")) {
      $("fighter1").value = "Islam Makhachev";
      $("fighter2").value = "Dustin Poirier";
      form.querySelector('input[name="rounds"][value="5"]').checked = true;
      await analyze();
    }
  } catch (error) {
    $("connection").textContent = error.name === "TimeoutError" ? "Connection timed out. Check the local server and retry." : error.message;
    $("retry").hidden = false;
  }
}

form.addEventListener("submit", analyze);
form.addEventListener("input", () => {
  showError("");
  $("activity").textContent = "Matchup changed. Analyze again to update the results.";
  if (lastResult) {
    $("results").hidden = true;
    $("empty").hidden = false;
  }
});
$("retry").addEventListener("click", loadFighters);
loadFighters();
