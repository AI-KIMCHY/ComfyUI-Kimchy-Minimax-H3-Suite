import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "Kimchy.H3UpscalerMP.AutoToggle",
    async nodeCreated(node) {
        if (node.comfyClass !== "KimchyH3ZeroSplitMP") return;

        const modeWidget = node.widgets?.find((w) => w.name === "upscale_by");
        const mpWidget = node.widgets?.find((w) => w.name === "target_megapixels");
        const scaleWidget = node.widgets?.find((w) => w.name === "scale");

        if (!modeWidget || !mpWidget || !scaleWidget) return;

        const toggleWidget = (widget, show) => {
            if (!widget._origType) {
                widget._origType = widget.type;
                widget._origComputeSize = widget.computeSize;
            }
            widget.type = show ? widget._origType : "converted-widget";
            widget.computeSize = show ? widget._origComputeSize : () => [0, -4];
            widget.disabled = !show;
        };

        const updateVisibility = () => {
            const isMP = modeWidget.value === "megapixels";
            toggleWidget(mpWidget, isMP);
            toggleWidget(scaleWidget, !isMP);
            node.setSize([node.size[0], node.computeSize()[1]]);
            app.graph.setDirtyCanvas(true, true);
        };

        const origCallback = modeWidget.callback;
        modeWidget.callback = function (...args) {
            if (origCallback) origCallback.apply(this, args);
            updateVisibility();
        };

        setTimeout(updateVisibility, 50);
    },
});
