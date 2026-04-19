import { Link, Navigate, Route, Routes } from "react-router-dom";
import type { ReactNode } from "react";
import { HomePage, ProjectsPage, ProjectWorkspace } from "./pages";

function Layout({ children }: { children: ReactNode }) {
  return (
    <div>
      <header
        style={{
          borderBottom: "1px solid var(--border)",
          background: "rgba(26,35,50,0.85)",
          backdropFilter: "blur(8px)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        <div
          className="container"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
            paddingTop: "0.85rem",
            paddingBottom: "0.85rem",
          }}
        >
          <Link to="/" style={{ fontWeight: 700, fontSize: "1.15rem" }}>
            GeneMiner
          </Link>
          <nav style={{ display: "flex", gap: "1.25rem" }}>
            <Link to="/">Overview</Link>
            <Link to="/projects">Projects</Link>
          </nav>
        </div>
      </header>
      <div className="container">{children}</div>
    </div>
  );
}

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:id" element={<ProjectWorkspace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
