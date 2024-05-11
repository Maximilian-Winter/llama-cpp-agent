import re


class TextChunker:
    def __init__(self, text, chunk_size, overlap=0):
        """
        Initializes the TextChunker instance.

        Parameters:
            text (str): The text to be chunked.
            chunk_size (int): The length of each text chunk.
            overlap (int): The number of characters that should overlap between consecutive chunks.
        """
        self.text = text
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_chunks(self):
        """
        Generates the text chunks based on the specified chunk size and overlap.

        Returns:
            list[str]: A list of chunked text segments.
        """
        chunks = []
        start = 0
        while start < len(self.text):
            # Calculate end index of the chunk
            end = start + self.chunk_size
            # Append chunk to the list
            chunks.append(self.text[start:end])
            # Update start index for the next chunk
            start = end - self.overlap if self.overlap < self.chunk_size else start + 1
            # Prevent infinite loop in case overlap is not less than chunk_size
            if self.chunk_size <= self.overlap:
                raise ValueError(
                    "Overlap must be less than chunk size to prevent infinite loops."
                )
        return chunks


class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        separators,
        chunk_size,
        chunk_overlap,
        length_function=len,
        keep_separator=False,
    ):
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.keep_separator = keep_separator

    def split_text(self, text, depth=0):
        if depth == len(self.separators):
            return self._split_into_fixed_size(text)

        current_separator = self.separators[depth]
        if current_separator == "":
            return list(text)

        if self.keep_separator:
            # Use regex to keep separators with the text
            pieces = re.split(f"({re.escape(current_separator)})", text)
            # Reattach separators to the chunks
            pieces = [
                (
                    pieces[i] + pieces[i + 1]
                    if i + 1 < len(pieces) and pieces[i + 1] == current_separator
                    else pieces[i]
                )
                for i in range(0, len(pieces), 2)
            ]
        else:
            pieces = text.split(current_separator)

        refined_pieces = []

        for piece in pieces:
            if self.length_function(piece) > self.chunk_size:
                refined_pieces.extend(self.split_text(piece, depth + 1))
            else:
                refined_pieces.append(piece)

        return self._merge_pieces(refined_pieces) if depth == 0 else refined_pieces

    def _split_into_fixed_size(self, text):
        size = self.chunk_size
        overlap = self.chunk_overlap
        chunks = [text[i : i + size] for i in range(0, len(text), size - overlap)]
        if chunks and len(chunks[-1]) < overlap:
            chunks[-2] += chunks[-1]
            chunks.pop()
        return chunks

    def _merge_pieces(self, pieces):
        merged = []
        current_chunk = pieces[0]

        for piece in pieces[1:]:
            if self.length_function(current_chunk + piece) <= self.chunk_size:
                current_chunk += piece
            else:
                merged.append(current_chunk)
                if len(current_chunk) == self.chunk_size:
                    current_chunk = current_chunk[-self.chunk_overlap :] + piece
                else:
                    current_chunk = piece

        merged.append(current_chunk)
        return merged
