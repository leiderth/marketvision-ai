/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        mv: {
          bg: "#0a0e14",
          panel: "#10151d",
          border: "#1f2733",
          up: "#2ecc71",
          down: "#e74c3c",
          neutral: "#8a97a8",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
