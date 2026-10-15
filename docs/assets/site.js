fetch("metrics.json").then(response => response.ok ? response.json() : Promise.reject()).then(metrics => {
  document.querySelectorAll("[data-metric]").forEach(element => {
    const value = metrics[element.dataset.metric];
    if (value === undefined) return;
    element.textContent = element.dataset.metric.endsWith("rate")
      ? `${(value * 100).toFixed(1)}%`
      : Number(value).toLocaleString();
  });
}).catch(() => {});