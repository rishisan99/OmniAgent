import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    // Docker can override with BACKEND_INTERNAL_URL=http://backend:8000.
    const backend = process.env.BACKEND_INTERNAL_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backend}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
