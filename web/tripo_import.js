import { app } from "../../scripts/app.js";

const MODEL_EXTENSIONS = ".glb,.fbx,.obj,.stl,.ply,.3mf,.gltf";

app.registerExtension({
    name: "TripoAPI.ImportModel",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "TripoImportModel") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            // Find the model_file combo widget
            const widget = this.widgets?.find(w => w.name === "model_file");
            if (!widget) return;

            // Add upload button after the combo widget
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = MODEL_EXTENSIONS;
            fileInput.style.display = "none";
            document.body.appendChild(fileInput);

            const uploadBtn = this.addWidget("button", "upload_model", "Upload Model File", () => {
                fileInput.click();
            });
            uploadBtn.label = "Upload Model File";

            fileInput.onchange = async () => {
                const file = fileInput.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append("image", file, file.name);
                formData.append("type", "input");
                formData.append("overwrite", "false");

                try {
                    const resp = await fetch("/upload/image", {
                        method: "POST",
                        body: formData,
                    });
                    if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
                    const data = await resp.json();
                    const uploaded = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;

                    // Refresh the combo options and select the uploaded file
                    widget.options.values = [...(widget.options.values || [])];
                    if (!widget.options.values.includes(uploaded)) {
                        widget.options.values.push(uploaded);
                    }
                    widget.value = uploaded;
                    app.graph.setDirtyCanvas(true);
                } catch (e) {
                    alert("Upload failed: " + e.message);
                }

                fileInput.value = "";
            };
        };
    },
});
