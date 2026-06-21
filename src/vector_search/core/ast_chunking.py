import tree_sitter
import tree_sitter_python as tspython


class CASTChunker:
    def __init__(self, max_chunk_size=1200):  # reduced from 2500 to avoid merging functions
        self.max_chunk_size = max_chunk_size
        language_capsule = tspython.language()
        self.language = tree_sitter.Language(language_capsule)
        self.parser = tree_sitter.Parser()
        self.parser.language = self.language

    def get_size(self, node):
        """Number of non-whitespace characters."""
        text = node.text.decode("utf-8")
        return len("".join(text.split()))

    def chunk_code(self, source_code, file_path=""):
        """Returns a list of strings (same as before) so nothing else breaks.
        Internally now chunks at function/class level and enriches with docstrings.
        file_path is optional — pass it if you want source metadata stored.
        """
        self._current_file = file_path
        tree = self.parser.parse(bytes(source_code, "utf8"))
        root_node = tree.root_node

        # If the whole file fits, return it as one chunk
        if self.get_size(root_node) <= self.max_chunk_size:
            return [root_node.text.decode("utf-8")]

        # Otherwise extract meaningful nodes
        chunks = []
        self._extract_meaningful_nodes(root_node, chunks)
        return chunks if chunks else self.chunk_nodes(root_node.children)

    def chunk_nodes(self, nodes):
        """Fallback: original algorithm for non-function/class nodes."""
        chunks = []
        current_chunk_nodes = []
        current_size = 0

        for node in nodes:
            node_size = self.get_size(node)
            if current_size + node_size > self.max_chunk_size:
                if current_chunk_nodes:
                    chunks.append(self._nodes_to_text(current_chunk_nodes))
                    current_chunk_nodes = []
                    current_size = 0
                if node_size > self.max_chunk_size:
                    sub_chunks = self.chunk_nodes(node.children)
                    chunks.extend(sub_chunks)
                    continue
            current_chunk_nodes.append(node)
            current_size += node_size

        if current_chunk_nodes:
            chunks.append(self._nodes_to_text(current_chunk_nodes))
        return chunks

    def _nodes_to_text(self, nodes):
        """Helper to combine a list of AST nodes back into a string."""
        return "\n".join([n.text.decode("utf-8") for n in nodes])

    # ------------------------------------------------------------------
    # New internals
    # ------------------------------------------------------------------

    def _extract_meaningful_nodes(self, node, chunks: list, parent_class: str = ""):
        """Recursively walk tree, emit one enriched chunk per function/class.

        ``parent_class`` is threaded through so that methods inside a class
        that had to be split (too large) still carry the class name.
        """
        if node.type == "class_definition":
            class_name = self._get_node_name(node)
            size = self.get_size(node)
            if size <= self.max_chunk_size:
                chunks.append(self._enrich_node(node, parent_class=""))
                return
            # Class too big — recurse into children, passing class_name down
            for child in node.children:
                self._extract_meaningful_nodes(child, chunks, parent_class=class_name)
        elif node.type == "function_definition":
            size = self.get_size(node)
            if size <= self.max_chunk_size:
                chunks.append(self._enrich_node(node, parent_class=parent_class))
                return
            for child in node.children:
                self._extract_meaningful_nodes(child, chunks, parent_class=parent_class)
        else:
            for child in node.children:
                self._extract_meaningful_nodes(child, chunks, parent_class=parent_class)

    def _enrich_node(self, node, parent_class: str = "") -> str:
        """Return enriched chunk string with class context, name, docstring, and source."""
        name = self._get_node_name(node)
        docstring = self._extract_docstring(node)
        raw_text = node.text.decode("utf-8")

        enriched = ""
        if parent_class:
            enriched += f"# class: {parent_class}\n"
        if name:
            enriched += f"# {name}\n"
        if docstring:
            enriched += f"# {docstring[:300]}\n\n"
        enriched += raw_text
        return enriched

    def _get_node_name(self, node) -> str:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8")
        return ""

    def _extract_docstring(self, node) -> str:
        """Extract first string literal from function/class body."""
        for child in node.children:
            if child.type == "block":
                for block_child in child.children:
                    if block_child.type == "expression_statement":
                        for expr_child in block_child.children:
                            if expr_child.type == "string":
                                raw = expr_child.text.decode("utf-8")
                                return raw.strip('"""').strip("'''").strip('"').strip("'").strip()
        return ""
