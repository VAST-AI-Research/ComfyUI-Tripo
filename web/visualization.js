import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const min_width = 600, min_height = 500;

class Visualizer {
  constructor(node, container, visualSrc) {
    this.node = node;

    this.iframe = document.createElement("iframe");
    Object.assign(this.iframe, {
      scrolling: "no",
      overflow: "hidden",
    });
    this.iframe.src =
      "/extensions/ComfyUI-Tripo/html/" + visualSrc + ".html";
    container.appendChild(this.iframe);
  }

  updateVisual(params) {
    this.model_viewer.src = api
      .apiURL("/view?" + new URLSearchParams(params))
      .replace(/extensions.*\//, "");
    this.model_viewer.setAttribute("camera-orbit", "90deg 90deg 100%");
  }

  remove() {
    this.container.remove();
  }
}

function createVisualizer(node, inputName, typeName, inputData, app) {
  node.name = inputName;

  const widget = {
    type: typeName,
    name: "preview3d",
    callback: () => {},
    model_viewer: null,
    model_div: null,
    draw: function (ctx, node, widgetWidth, widgetY, widgetHeight) {
      const margin = 10;
      const top_offset = 5;
      const visible = app.canvas.ds.scale > 0.5 && this.type === typeName;
      const w = widgetWidth - margin * 4;
      const clientRectBound = ctx.canvas.getBoundingClientRect();
      const transform = new DOMMatrix()
        .scaleSelf(
          clientRectBound.width / ctx.canvas.width,
          clientRectBound.height / ctx.canvas.height
        )
        .multiplySelf(ctx.getTransform())
        .translateSelf(margin, margin + widgetY);
      Object.assign(this.visualizer.style, {
        left: `${transform.a * margin + transform.e}px`,
        top: `${transform.d + transform.f + top_offset}px`,
        width: `${w * transform.a}px`,
        height: `${
          w * transform.d - widgetHeight - margin * 15 * transform.d
        }px`,
        position: "absolute",
        overflow: "hidden",
        zIndex: app.graph._nodes.indexOf(node),
      });

      Object.assign(this.visualizer.children[0].style, {
        transformOrigin: "50% 50%",
        width: "100%",
        height: "100%",
        border: "0 none",
      });
      if (this.model_viewer != null) {
        this.model_viewer.style.width = this.visualizer.style.width;
        this.model_viewer.style.height = this.visualizer.style.height;
        console.log(this.model_viewer.getCameraOrbit());
      }
      this.visualizer.hidden = !visible;
    },
  };

  const container = document.createElement("div");
  container.id = `ComfyTripoApi_${inputName}`;

  node.visualizer = new Visualizer(node, container, typeName);
  widget.visualizer = container;
  widget.parent = node;
  node.visualizer.iframe.onload = () => {
    const iframeDocument = node.visualizer.iframe.contentWindow.document;
    node.visualizer.model_viewer = iframeDocument.getElementById("model-viewer");
    widget.model_viewer = node.visualizer.model_viewer;
    iframeDocument.body.style.margin = "0";
  };
  document.body.appendChild(widget.visualizer);

  node.addCustomWidget(widget);

  node.updateParameters = (params) => {
    node.visualizer.updateVisual(params);
  };

  // Events for drawing backgound
  node.onDrawBackground = function (ctx) {
    if (!this.flags.collapsed) {
      node.visualizer.iframe.hidden = false;
    } else {
      node.visualizer.iframe.hidden = true;
    }
  };

  // Make sure visualization iframe is always inside the node when resize the node
  node.onResize = function () {
    let [w, h] = this.size;
    w = Math.max(w, min_width);
    h = Math.max(h, min_height);


    if (w > min_width) {
      h = w - 100;
    }
    this.size = [w, h];
  };

  // Events for remove nodes
  node.onRemoved = () => {
    for (let w in node.widgets) {
      if (node.widgets[w].visualizer) {
        node.widgets[w].visualizer.remove();
      }
    }
  };

  return {
    widget: widget,
  };
}

function registerVisualizer(nodeType, nodeData, nodeClassName, typeName) {
  if (nodeData.name == nodeClassName) {
    console.log("[3D Visualizer] Registering node: " + nodeData.name);

    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = async function () {
      const r = onNodeCreated
        ? onNodeCreated.apply(this, arguments)
        : undefined;

      let Preview3DNode = app.graph._nodes.filter(
        (wi) => wi.type == nodeClassName
      );
      let nodeName = `Preview3DNode_${Preview3DNode.length}`;

      console.log(`[Tripo] Create: ${nodeName}`);

      const result = await createVisualizer.apply(this, [
        this,
        nodeName,
        typeName,
        {},
        app,
      ]);

      this.setSize([min_width, min_height]);

      return r;
    };

    nodeType.prototype.onExecuted = async function (message) {
      if (message?.mesh) {
        this.updateParameters(message.mesh[0]);
      }
    };
  }
}

app.registerExtension({
  name: "jw782cn",

  async init(app) {},

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    registerVisualizer(nodeType, nodeData, "TripoGLBViewer", "model-viewer");
  },
});
