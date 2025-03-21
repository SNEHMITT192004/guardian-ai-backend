/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg': 'rgba(218,218,218,0.40)',
        'screen': '#4d4d4d',
        'chatbot': '#949494'
      },
      backgroundImage: {
        'url': 'linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url(assets/bg.jpg)'
      }
    },
  },
  plugins: [],
}