import React, { useRef, useState } from "react";
import { createPortal } from "react-dom";
import "./InfoTip.css";

// Info tooltip button. Shows explanation on hover or keyboard focus via a
// portal so it is not clipped by parent overflow. Prop `text` is the
// explanation.

export default function InfoTip({ text, ariaLabel = "More info" }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState(null);
  const btnRef = useRef(null);

  const openAt = () => {
    const el = btnRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const tipWidth = 260;
    const vw = window.innerWidth;
    // Prefer right of the icon, clamp so we don't overflow the viewport.
    let left = rect.right + 8;
    if (left + tipWidth > vw - 12) {
      left = Math.max(12, rect.left - tipWidth - 8);
    }
    const top = rect.top - 4;
    setPos({ top, left, width: tipWidth });
    setOpen(true);
  };
  const close = () => setOpen(false);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        className="info-tip-btn"
        aria-label={ariaLabel}
        onMouseEnter={openAt}
        onMouseLeave={close}
        onFocus={openAt}
        onBlur={close}
        onClick={(e) => {
          e.preventDefault();
          open ? close() : openAt();
        }}
      >
        i
      </button>
      {open && pos && createPortal(
        <div
          role="tooltip"
          className="info-tip-bubble"
          style={{ top: pos.top, left: pos.left, width: pos.width }}
        >
          {text}
        </div>,
        document.body,
      )}
    </>
  );
}
