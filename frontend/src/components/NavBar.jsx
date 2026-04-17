import React from "react";
import { NavLink } from "react-router-dom";
import "./NavBar.css";

// Single consistent nav across every page. Uses NavLink so the active page
// gets an underline / bold style automatically. Kept as a flat row of text
// links so it doesn't compete with page content for attention.

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/demos", label: "Live Demos" },
  { to: "/gallery", label: "Gallery" },
  { to: "/reports", label: "Reports" },
  { to: "/docs", label: "Docs" },
];

export default function NavBar() {
  return (
    <nav className="app-navbar" aria-label="Primary">
      <div className="app-navbar-inner">
        <NavLink to="/" className="app-navbar-brand" end>
          Gaussian Splatting
        </NavLink>
        <ul className="app-navbar-links">
          {LINKS.map(({ to, label, end }) => (
            <li key={to}>
              <NavLink
                to={to}
                end={end}
                className={({ isActive }) =>
                  "app-navbar-link" + (isActive ? " is-active" : "")
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
