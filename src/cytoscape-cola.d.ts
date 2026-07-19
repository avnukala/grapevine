// cytoscape-cola ships no types; it's a Cytoscape extension registrar.
declare module "cytoscape-cola" {
  const ext: (cytoscape: unknown) => void;
  export = ext;
}
