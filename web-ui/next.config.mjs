/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  output: "standalone",
  webpack: (config) => {
    config.resolve = config.resolve || {};
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      encoding: false,
      'pino-pretty': false,
    };
    return config;
  },
  async rewrites() {
    return [
      {
        source: '/api/backend/:path*',
        destination: 'http://state-service:8000/:path*',
      },
      {
        source: '/api/tester/:path*',
        destination: 'http://solution-tester:8008/:path*',
      },
      {
        source: '/api/policy/:path*',
        destination: 'http://policy-models:8001/:path*',
      },
      {
        source: '/api/turnkey/:path*',
        destination: 'https://auth.turnkey.com/:path*',
      },
      {
        source: '/api/alchemy/:path*',
        destination: 'https://api.g.alchemy.com/:path*',
      },
    ];
  },
};

export default nextConfig;
