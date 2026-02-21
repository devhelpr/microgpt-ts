/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#070b14',
        slate: '#101a31',
        neon: '#44f2d9',
        coral: '#ff7e6b',
        butter: '#ffe08c',
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(68, 242, 217, 0.2), 0 30px 80px rgba(68, 242, 217, 0.15)',
      },
      animation: {
        pulseSoft: 'pulseSoft 2.2s ease-in-out infinite',
        floatSlow: 'floatSlow 8s ease-in-out infinite',
      },
      keyframes: {
        pulseSoft: {
          '0%, 100%': { opacity: '0.7' },
          '50%': { opacity: '1' },
        },
        floatSlow: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-8px)' },
        },
      },
    },
  },
  plugins: [],
};
