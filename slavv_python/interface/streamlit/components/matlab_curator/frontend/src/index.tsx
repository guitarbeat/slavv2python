import { createRoot, type Root } from "react-dom/client";
import type { FrontendRenderer } from "@streamlit/component-v2-lib";

import { App } from "./App";
import { decodeCuratorPayload } from "./payload";
import "./style.css";

const roots = new WeakMap<HTMLElement | ShadowRoot, Root>();

const renderer: FrontendRenderer = ({ parentElement, data, setTriggerValue }) => {
  let root = roots.get(parentElement);
  if (!root) {
    root = createRoot(parentElement);
    roots.set(parentElement, root);
  }
  root.render(
    <App
      data={decodeCuratorPayload(data)}
      setTriggerValue={(name, value) => setTriggerValue(name, value)}
    />,
  );
};

export default renderer;
