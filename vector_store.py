"""
PRODUCTION-READY Vector Store for Iron Lady Leadership Program
================================================================
✅ Multi-format: .docx, .md, .pdf
✅ Table preservation with markdown conversion
✅ Optimal chunking for detailed frameworks
✅ Section-aware metadata
✅ OpenAI embeddings for best RAG

Author: Optimized for 100BM RAG System
Date: November 2025
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,  # Better table handling than Docx2txt
    UnstructuredMarkdownLoader,
    PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()


# ============================================================================
# METADATA EXTRACTORS
# ============================================================================

class MetadataExtractor:
    """Extract rich metadata from file names and paths"""
    
    def __init__(self):
        # YouTube URL patterns
        self.youtube_patterns = [
            re.compile(r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', re.IGNORECASE),
            re.compile(r'(?:youtube|utube|video|link|url)\s*(?:url)?\s*[:=-]\s*(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', re.IGNORECASE),
        ]
    
    def extract_youtube_urls(self, text: str) -> List[str]:
        """Extract YouTube URLs from text"""
        video_ids = set()
        for pattern in self.youtube_patterns:
            matches = pattern.findall(text)
            video_ids.update(matches)
        return [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
    
    @staticmethod
    def extract_session_number(file_path: str) -> Optional[int]:
        """Extract session number from path or filename"""
        patterns = [
            r'Session\s*(\d+)',
            r'session\s*(\d+)',
            r'Session-(\d+)',
            r'Core Session (\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, file_path, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    @staticmethod
    def extract_facilitator(filename: str) -> Optional[str]:
        """Extract facilitator name from filename"""
        patterns = [
            r'by\s+([A-Z][a-zA-Z\s]+?)(?:\.|$)',
            r'-\s*([A-Z][a-zA-Z\s]+?)\.(?:docx|md|pdf)',
            r'Showcase\s*-\s*([A-Z][a-zA-Z\s]+)',
            r'Facilitator:\s*([A-Z][a-zA-Z\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                name = match.group(1).strip()
                if name not in ['Session', 'Image', 'Iron', 'Lady', 'Way', 'Complete', 'Knowledge', 'Base']:
                    return name
        
        return None
    
    @staticmethod
    def get_content_type(file_path: str) -> str:
        """Determine content type based on path"""
        path_lower = file_path.lower()
        
        # Core sessions
        if 'session 1' in path_lower or 'image creation' in path_lower:
            return 'session_1_image_creation'
        elif 'session 2' in path_lower or '4t management' in path_lower:
            return 'session_2_4t_management'
        elif 'session 3' in path_lower or 'capability' in path_lower:
            return 'session_3_capability'
        elif 'session 4' in path_lower or 'pitch' in path_lower:
            return 'session_4_pitch'
        elif 'session 5' in path_lower or 'bing fa' in path_lower or 'stratagem' in path_lower:
            return 'session_5_strategy'
        elif 'session 6' in path_lower:
            return 'session_6_implementation'
        
        # Special categories
        elif 'community' in path_lower:
            return 'community_session'
        elif 'revision' in path_lower:
            return 'revision_session'
        elif 'boardroom' in path_lower:
            return 'boardroom_session'
        
        return 'general'
    
    @staticmethod
    def get_session_title(file_path: str, session_num: Optional[int]) -> str:
        """Get descriptive session title"""
        titles = {
            1: "Image Creation - Iron Lady Way",
            2: "4T Management for Operational Excellence",
            3: "Breakthrough Capability Development",
            4: "Pitch Without Pitching & Influencing Tactics",
            5: "Bing Fa Stratagem (11-Point Framework)",
            6: "Real-Time Implementation & Board Tips"
        }
        
        if session_num and session_num in titles:
            return titles[session_num]
        
        # Extract from filename
        filename = Path(file_path).name
        name = re.sub(r'\.(docx|md|pdf)$', '', filename, flags=re.IGNORECASE)
        
        if '-' in name:
            parts = name.split('-', 1)
            if len(parts) > 1:
                return parts[1].strip()
        
        return name.strip()
    
    @staticmethod
    def detect_tables_in_content(text: str) -> bool:
        """Detect if content contains markdown tables"""
        # Look for markdown table patterns
        table_pattern = r'\|[^\n]+\|'
        return bool(re.search(table_pattern, text))


# ============================================================================
# TABLE PROCESSOR
# ============================================================================

class TableProcessor:
    """Preserve table structure in chunks"""
    
    @staticmethod
    def extract_tables(text: str) -> List[Dict[str, Any]]:
        """Extract markdown tables and their positions"""
        tables = []
        
        # Pattern for markdown tables
        # Matches: | col1 | col2 |\n|---|---|\n| val1 | val2 |
        table_pattern = r'(\|[^\n]+\|\n\|[\s\-:|]+\|\n(?:\|[^\n]+\|\n?)+)'
        
        for match in re.finditer(table_pattern, text):
            tables.append({
                'content': match.group(0),
                'start': match.start(),
                'end': match.end()
            })
        
        return tables
    
    @staticmethod
    def mark_table_boundaries(text: str) -> str:
        """Add markers around tables to prevent splitting"""
        tables = TableProcessor.extract_tables(text)
        
        if not tables:
            return text
        
        # Add markers from end to start (to preserve positions)
        result = text
        for table in reversed(tables):
            table_content = table['content']
            marked_table = f"\n\n[TABLE_START]\n{table_content}\n[TABLE_END]\n\n"
            result = result[:table['start']] + marked_table + result[table['end']:]
        
        return result


# ============================================================================
# ADVANCED DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    """Process documents with intelligent chunking and metadata"""
    
    def __init__(self):
        self.metadata_extractor = MetadataExtractor()
        self.table_processor = TableProcessor()
        
        # OPTIMIZED text splitter for your detailed knowledge base
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # LARGER for detailed frameworks and examples
            chunk_overlap=400,  # MORE overlap to preserve context
            length_function=len,
            separators=[
                "\n\n## SECTION",      # Main sections
                "\n\n### ",             # Subsections
                "\n\n#### ",            # Sub-subsections
                "\n\n**",               # Bold headers
                "[TABLE_START]",        # Table boundaries (don't split)
                "\n\n",                 # Paragraphs
                "\n",                   # Lines
                ". ",                   # Sentences
                " ",                    # Words
                ""                      # Characters
            ]
        )
        
        # Special splitter for table-heavy content
        self.table_aware_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,  # Even larger to keep tables intact
            chunk_overlap=500,
            length_function=len,
            separators=[
                "[TABLE_START]",
                "\n\n## SECTION",
                "\n\n### ",
                "\n\n",
                "\n",
            ]
        )
    
    def extract_section_info(self, text: str, chunk: str) -> Dict[str, Optional[str]]:
        """Extract section information from structured content"""
        section_info = {
            'section_number': None,
            'section_title': None,
            'subsection': None
        }
        
        # Find chunk position
        chunk_start = chunk[:100].strip()
        chunk_pos = text.find(chunk_start)
        
        if chunk_pos == -1:
            # Try with cleaned chunk (remove table markers)
            clean_chunk = chunk.replace('[TABLE_START]', '').replace('[TABLE_END]', '')
            chunk_start = clean_chunk[:100].strip()
            chunk_pos = text.find(chunk_start)
        
        if chunk_pos != -1:
            text_before = text[:chunk_pos + 100]
            
            # Section patterns
            section_patterns = [
                r'##\s*SECTION\s+(\d+):\s*([^\n]+)',
                r'SECTION\s+(\d+):\s*([^\n]+)',
                r'##\s*(\d+)\.\s*([^\n]+)',
            ]
            
            for pattern in section_patterns:
                matches = list(re.finditer(pattern, text_before, re.IGNORECASE))
                if matches:
                    last = matches[-1]
                    section_info['section_number'] = int(last.group(1))
                    section_info['section_title'] = last.group(2).strip()
                    break
            
            # Subsection patterns
            subsection_patterns = [
                r'###\s+([^\n]+)',
                r'####\s+([^\n]+)',
                r'\*\*([^*]+)\*\*:',
            ]
            
            for pattern in subsection_patterns:
                matches = list(re.finditer(pattern, text_before))
                if matches:
                    section_info['subsection'] = matches[-1].group(1).strip()
                    break
        
        return section_info
    
    def load_document(self, file_path: Path) -> List[Document]:
        """Load document with format-specific handling"""
        try:
            suffix = file_path.suffix.lower()
            
            # Select loader based on format
            if suffix == '.docx':
                # UnstructuredWordDocumentLoader preserves tables better
                loader = UnstructuredWordDocumentLoader(
                    str(file_path),
                    mode="elements"  # Preserves structure
                )
            elif suffix == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
            else:
                print(f"  ⚠️  Unsupported format: {suffix}")
                return []
            
            # Load content
            raw_docs = loader.load()
            
            if not raw_docs:
                return []
            
            # Combine all content
            full_content = "\n\n".join([doc.page_content for doc in raw_docs])
            
            if not full_content.strip():
                return []
            
            # Detect and mark tables
            has_tables = self.metadata_extractor.detect_tables_in_content(full_content)
            
            if has_tables:
                print(f"      📊 Tables detected - using table-aware chunking")
                full_content = self.table_processor.mark_table_boundaries(full_content)
                splitter = self.table_aware_splitter
            else:
                splitter = self.text_splitter
            
            # Extract metadata
            session_num = self.metadata_extractor.extract_session_number(str(file_path))
            facilitator = self.metadata_extractor.extract_facilitator(file_path.name)
            content_type = self.metadata_extractor.get_content_type(str(file_path))
            session_title = self.metadata_extractor.get_session_title(str(file_path), session_num)
            youtube_urls = self.metadata_extractor.extract_youtube_urls(full_content)
            
            parent_folder = file_path.parent.name
            
            # Split into chunks
            chunks = splitter.split_text(full_content)
            
            # Clean table markers from chunks
            chunks = [c.replace('[TABLE_START]', '').replace('[TABLE_END]', '') for c in chunks]
            
            processed_docs = []
            
            for idx, chunk in enumerate(chunks):
                section_info = self.extract_section_info(full_content, chunk)
                
                # Rich metadata (ChromaDB only accepts: str, int, float, bool, None)
                metadata = {
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'file_type': suffix[1:],
                    'parent_folder': parent_folder,
                    'session_number': session_num,
                    'session_title': session_title,
                    'facilitator': facilitator,
                    'content_type': content_type,
                    'has_tables': has_tables,
                    # Convert list to comma-separated string for ChromaDB compatibility
                    'youtube_urls': ', '.join(youtube_urls) if youtube_urls else None,
                    'processed_date': datetime.now().isoformat(),
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    
                    # Section-aware
                    'document_section_number': section_info['section_number'],
                    'document_section_title': section_info['section_title'],
                    'document_subsection': section_info['subsection'],
                }
                
                # Category tagging
                if 'community' in parent_folder.lower():
                    metadata['category'] = 'community'
                elif 'revision' in parent_folder.lower():
                    metadata['category'] = 'revision'
                elif session_num:
                    metadata['category'] = f'session_{session_num}'
                else:
                    metadata['category'] = 'general'
                
                # Enhanced content with context
                content_prefix = []
                
                if session_num:
                    content_prefix.append(f"📚 Session {session_num}: {session_title}")
                
                if facilitator:
                    content_prefix.append(f"👤 Facilitator: {facilitator}")
                
                if section_info['section_title']:
                    content_prefix.append(f"📖 Section {section_info['section_number']}: {section_info['section_title']}")
                
                if section_info['subsection']:
                    content_prefix.append(f"📌 Topic: {section_info['subsection']}")
                
                if has_tables:
                    content_prefix.append(f"📊 Contains structured tables/data")
                
                content_prefix.append(f"📄 Source: {file_path.name}")
                content_prefix.append("---")
                
                enhanced_content = "\n".join(content_prefix) + "\n\n" + chunk
                
                doc = Document(
                    page_content=enhanced_content,
                    metadata=metadata
                )
                processed_docs.append(doc)
            
            return processed_docs
            
        except Exception as e:
            print(f"  ⚠️  Error loading {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return []


# ============================================================================
# VECTOR STORE CREATOR
# ============================================================================

class VectorStoreCreator:
    """Create optimized vector store from structured content"""
    
    def __init__(
        self,
        content_folder: str = "./lms_content",
        vector_store_path: str = "./vector_store",
        embedding_model: str = "openai"
    ):
        self.content_folder = Path(content_folder)
        self.vector_store_path = vector_store_path
        
        # Setup embeddings
        if embedding_model == "openai":
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large", 
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError("Only OpenAI embeddings supported")
        
        self.processor = DocumentProcessor()
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'total_documents': 0,
            'by_category': {},
            'by_session': {},
            'by_format': {},
            'files_with_tables': 0,
            'facilitators': set()
        }
    
    def discover_files(self) -> List[Path]:
        """Discover all supported files"""
        print(f"\n📂 Discovering files in: {self.content_folder}")
        
        if not self.content_folder.exists():
            print(f"❌ Error: Folder '{self.content_folder}' not found!")
            return []
        
        # Find all supported files
        files = []
        for ext in ['*.docx', '*.md', '*.pdf']:
            found = list(self.content_folder.glob(f"**/{ext}"))
            files.extend(found)
            print(f"   Found {len(found)} {ext} files")
        
        # Filter temp files
        files = [f for f in files if not f.name.startswith(('~', '.'))]
        
        print(f"\n✓ Total: {len(files)} files")
        
        # Show distribution
        formats = {}
        folders = {}
        
        for file in files:
            fmt = file.suffix.lower()
            formats[fmt] = formats.get(fmt, 0) + 1
            
            folder = file.parent.name
            folders[folder] = folders.get(folder, 0) + 1
        
        print("\n📊 By Format:")
        for fmt, count in sorted(formats.items()):
            print(f"   • {fmt}: {count} files")
        
        print("\n📂 By Folder:")
        for folder, count in sorted(folders.items()):
            print(f"   • {folder}: {count} files")
        
        return files
    
    def load_all_documents(self, files: List[Path]) -> List[Document]:
        """Load all documents"""
        print(f"\n📖 Loading and processing documents...")
        print("="*60)
        
        all_documents = []
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {file_path.name}")
            print(f"   📁 {file_path.parent.name}/")
            
            docs = self.processor.load_document(file_path)
            
            if docs:
                all_documents.extend(docs)
                
                self.stats['total_files'] += 1
                
                # Format tracking
                fmt = file_path.suffix.lower()
                self.stats['by_format'][fmt] = self.stats['by_format'].get(fmt, 0) + 1
                
                # Table tracking
                if docs[0].metadata.get('has_tables'):
                    self.stats['files_with_tables'] += 1
                
                # Category tracking
                category = docs[0].metadata.get('category', 'unknown')
                self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + 1
                
                # Session tracking
                session = docs[0].metadata.get('session_number')
                if session:
                    self.stats['by_session'][session] = self.stats['by_session'].get(session, 0) + 1
                
                # Facilitator tracking
                facilitator = docs[0].metadata.get('facilitator')
                if facilitator:
                    self.stats['facilitators'].add(facilitator)
                
                print(f"   ✓ Created {len(docs)} chunks")
            else:
                print(f"   ⚠️  Failed to load")
        
        self.stats['total_documents'] = len(all_documents)
        
        print("\n" + "="*60)
        print(f"✓ Successfully loaded {len(all_documents)} chunks from {self.stats['total_files']} files")
        
        return all_documents
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create vector store with batch processing to avoid token limits"""
        print(f"\n🔧 Creating vector store...")
        print(f"   📍 Location: {self.vector_store_path}")
        print(f"   📚 Documents: {len(documents)}")
        print(f"   🤖 Embedding: OpenAI text-embedding-3-large")
        print(f"   ⏳ This may take a few minutes...")
        
        # Batch size to avoid OpenAI token limits (300K tokens per request)
        # Average chunk ~2000 tokens, so 100 chunks = ~200K tokens (safe margin)
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        print(f"\n📦 Processing in batches of {batch_size} documents...")
        print(f"   Total batches: {total_batches}")
        
        # Initialize vector store with first batch
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                first_batch = documents[:batch_size]
                vector_store = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=self.vector_store_path
                )
                print(f"   ✓ Batch 1/{total_batches} complete ({len(first_batch)} docs)")
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    if attempt < max_retries - 1:
                        print(f"   ⚠️  Rate limit hit, waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
                else:
                    raise
        
        # Add remaining batches
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            for attempt in range(max_retries):
                try:
                    vector_store.add_documents(batch)
                    print(f"   ✓ Batch {batch_num}/{total_batches} complete ({len(batch)} docs)")
                    
                    # Small delay between batches to avoid rate limits
                    if batch_num < total_batches:
                        time.sleep(0.5)
                    break
                    
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            print(f"   ⚠️  Rate limit hit on batch {batch_num}, waiting {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            print(f"   ❌ Failed batch {batch_num} after {max_retries} attempts")
                            raise
                    else:
                        print(f"   ❌ Error on batch {batch_num}: {e}")
                        raise
        
        count = vector_store._collection.count()
        print(f"\n✅ Vector store created with {count} vectors!")
        
        return vector_store
    
    def print_statistics(self):
        """Print comprehensive statistics"""
        print("\n" + "="*80)
        print("📊 VECTOR STORE STATISTICS")
        print("="*80)
        
        print(f"\n📁 Overview:")
        print(f"   • Files Processed: {self.stats['total_files']}")
        print(f"   • Total Chunks: {self.stats['total_documents']}")
        print(f"   • Files with Tables: {self.stats['files_with_tables']}")
        
        if self.stats['by_format']:
            print(f"\n📄 By Format:")
            for fmt, count in sorted(self.stats['by_format'].items()):
                print(f"   • {fmt}: {count} files")
        
        if self.stats['by_category']:
            print(f"\n📚 By Category:")
            for cat, count in sorted(self.stats['by_category'].items()):
                print(f"   • {cat}: {count} files")
        
        if self.stats['by_session']:
            print(f"\n🎓 By Session:")
            for sess in sorted(self.stats['by_session'].keys()):
                count = self.stats['by_session'][sess]
                print(f"   • Session {sess}: {count} files")
        
        if self.stats['facilitators']:
            print(f"\n👥 Facilitators:")
            for fac in sorted(self.stats['facilitators']):
                print(f"   • {fac}")
        
        print("\n" + "="*80)
    
    def run(self):
        """Execute pipeline"""
        print("="*80)
        print("🚀 IRON LADY LEADERSHIP - PRODUCTION VECTOR STORE")
        print("="*80)
        print("\n✨ Enhanced Features:")
        print("   ✅ Multi-format: .docx, .md, .pdf")
        print("   ✅ Table preservation with markdown")
        print("   ✅ Optimal chunking (2000 tokens, 400 overlap)")
        print("   ✅ Table-aware splitting (2500 tokens for tables)")
        print("   ✅ Section-aware metadata")
        print("   ✅ Session & facilitator extraction")
        print("   ✅ OpenAI embeddings (best quality)")
        print("="*80)
        
        # Check API key
        if not os.getenv("OPENAI_API_KEY"):
            print("\n❌ OPENAI_API_KEY not found in environment!")
            print("\n💡 Create .env file with:")
            print("   OPENAI_API_KEY=sk-your-key-here")
            return
        
        # Pipeline
        files = self.discover_files()
        if not files:
            print("\n❌ No files found!")
            print("\n💡 Make sure your content is in './lms_content' folder")
            return
        
        documents = self.load_all_documents(files)
        if not documents:
            print("\n❌ No documents loaded!")
            return
        
        self.create_vector_store(documents)
        self.print_statistics()
        
        print("\n🎉 SUCCESS! Vector store ready for RAG!")
        print("\n📝 Next Steps:")
        print("   1. Test with your RAG chatbot")
        print("   2. Query specific sessions using filters")
        print("   3. Tables and frameworks will be preserved")
        print("   4. Section context automatically included")
        print("\n" + "="*80)


# ============================================================================
# VECTOR STORE LOADER
# ============================================================================

class VectorStoreLoader:
    """Load and query vector store"""
    
    def __init__(self, vector_store_path: str = "./vector_store"):
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = None
    
    def load(self) -> Chroma:
        """Load vector store"""
        print(f"📂 Loading vector store from: {self.vector_store_path}")
        
        self.vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embeddings
        )
        
        count = self.vector_store._collection.count()
        print(f"✅ Loaded {count} vectors")
        
        return self.vector_store
    
    def search(self, query: str, k: int = 5, filter_dict: Optional[Dict] = None):
        """Search with optional filters"""
        if not self.vector_store:
            self.load()
        
        if filter_dict:
            return self.vector_store.similarity_search(query, k=k, filter=filter_dict)
        return self.vector_store.similarity_search(query, k=k)
    
    def search_by_session(self, query: str, session_number: int, k: int = 5):
        """Search within specific session"""
        return self.search(query, k=k, filter_dict={'session_number': session_number})
    
    def search_with_tables(self, query: str, k: int = 5):
        """Search only content with tables"""
        return self.search(query, k=k, filter_dict={'has_tables': True})


# ============================================================================
# MAIN
# ============================================================================

def main():
    creator = VectorStoreCreator(
        content_folder="./lms_content",
        vector_store_path="./vector_store",
        embedding_model="openai"
    )
    creator.run()


if __name__ == "__main__":
    main()
