import sys, types
if sys.platform == "win32":
    sys.modules["pwd"] = types.SimpleNamespace()
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import json

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# Load environment variables
load_dotenv()


class FolderBasedDocumentProcessor:
    """Process folder structure directly with enhanced URL detection"""
    
    def __init__(self):
        # Enhanced YouTube URL patterns
        self.youtube_patterns = [
            # Standard YouTube URLs
            re.compile(
                r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
                re.IGNORECASE
            ),
            # With labels like "Utube url:-", "YouTube:", "Video:", etc.
            re.compile(
                r'(?:utube|youtube|video|link|url)\s*(?:url)?\s*[:=-]\s*(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
                re.IGNORECASE
            ),
            # Just the youtube.com/watch part
            re.compile(
                r'youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
                re.IGNORECASE
            ),
            # youtu.be short links
            re.compile(
                r'youtu\.be/([a-zA-Z0-9_-]{11})',
                re.IGNORECASE
            ),
        ]
        
        # General URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
    
    def extract_youtube_urls(self, text: str) -> List[str]:
        """Extract YouTube URLs with enhanced detection"""
        video_ids = set()
        
        # Try all patterns
        for pattern in self.youtube_patterns:
            matches = pattern.findall(text)
            video_ids.update(matches)
        
        # Convert to full URLs
        youtube_urls = [f"https://www.youtube.com/watch?v={vid_id}" for vid_id in video_ids]
        return list(set(youtube_urls))
    
    def extract_all_urls(self, text: str) -> List[str]:
        """Extract all URLs"""
        return list(set(self.url_pattern.findall(text)))
    
    def extract_non_youtube_urls(self, text: str) -> List[str]:
        """Extract non-YouTube URLs"""
        all_urls = self.extract_all_urls(text)
        non_youtube = [url for url in all_urls if not any(yt in url.lower() for yt in ['youtube.com', 'youtu.be'])]
        return non_youtube
    
    def identify_section_from_path(self, file_path: Path) -> tuple:
        """
        IMPROVED: Identify section type from folder/file path
        Returns: (section_type, session_number, section_name, file_name)
        """
        parts = file_path.parts
        file_name = file_path.stem  # filename without extension
        
        # Get parent folder name
        if len(parts) > 1:
            parent_folder = parts[-2]
        else:
            parent_folder = ""
        
        # PRIORITY 1: Check if parent folder is exactly "100 BM Community"
        if parent_folder.strip().lower() == "100 bm community":
            return ('100bm_community', None, '100 BM Community', file_name)
        
        # PRIORITY 2: Check for explicit Session folder pattern
        session_folder_match = re.search(r'Session\s+(\d+)', parent_folder, re.IGNORECASE)
        if session_folder_match:
            session_num = int(session_folder_match.group(1))
            return ('session', session_num, parent_folder, file_name)
        
        # PRIORITY 3: Check for LEP revision
        if 'LEP' in parent_folder or 'LEP' in file_name:
            if 'revision' in parent_folder.lower() or 'revision' in file_name.lower():
                return ('lep_revision', None, 'LEP Revision', file_name)
        
        # PRIORITY 4: Check for Batch Schedule
        if 'batch' in parent_folder.lower() and 'schedule' in parent_folder.lower():
            return ('batch_schedule', None, 'Batch Schedule', file_name)
        
        if 'batch' in file_name.lower() and 'schedule' in file_name.lower():
            return ('batch_schedule', None, 'Batch Schedule', file_name)
        
        # PRIORITY 5: Check filename patterns for 100 BM Community files
        if re.match(r'^\d+\.\s*', file_name):
            return ('100bm_community', None, '100 BM Community', file_name)
        
        if 'boardroom' in file_name.lower():
            return ('100bm_community', None, '100 BM Community', file_name)
        
        if 'sawaal' in file_name.lower():
            return ('100bm_community', None, '100 BM Community', file_name)
        
        if 'industry leader' in file_name.lower():
            return ('100bm_community', None, '100 BM Community', file_name)
        
        # DEFAULT
        return ('100bm_community', None, '100 BM Community', file_name)
    
    def classify_content_type(self, text: str, youtube_count: int, other_url_count: int) -> str:
        """Classify content type"""
        total_urls = youtube_count + other_url_count
        
        if total_urls == 0:
            return 'content'
        
        word_count = len(text.split()) if text else 1
        url_density = total_urls / max(word_count / 10, 1)
        
        if url_density > 0.5:
            return 'url_list'
        elif total_urls > 0:
            return 'mixed'
        return 'content'


class FolderVectorStore:
    """Vector store that processes folder structure"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "100bm_community"
    ):
        # Load API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found. Check your .env file")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize processor
        self.doc_processor = FolderBasedDocumentProcessor()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.vectorstore = None
    
    def find_all_docx_files(self, root_path: str) -> List[Path]:
        """Find all .docx files in folder structure"""
        root = Path(root_path)
        docx_files = []
        
        # Recursively find all .docx files
        for file_path in root.rglob("*.docx"):
            # Skip temporary Word files (starting with ~$)
            if not file_path.name.startswith('~$'):
                docx_files.append(file_path)
        
        return sorted(docx_files)
    
    def process_folder_structure(self, root_path: str) -> List[Document]:
        """Process entire folder structure"""
        print(f"📁 Scanning folder: {root_path}")
        
        # Find all docx files
        docx_files = self.find_all_docx_files(root_path)
        
        print(f"📄 Found {len(docx_files)} .docx files")
        
        if not docx_files:
            raise ValueError(f"No .docx files found in {root_path}")
        
        # Process all files
        processed_docs = []
        total_youtube_urls = 0
        
        print(f"\n📊 Processing files and extracting YouTube URLs...")
        
        for file_path in docx_files:
            section_type, session_num, section_name, file_name = \
                self.doc_processor.identify_section_from_path(file_path)
            
            try:
                # Load document
                loader = Docx2txtLoader(str(file_path))
                raw_docs = loader.load()
                
                if not raw_docs or not raw_docs[0].page_content.strip():
                    print(f"   ⚠️  Skipping empty file: {file_path.name}")
                    continue
                
                content = raw_docs[0].page_content
                
                # Extract URLs from full content
                file_youtube_urls = self.doc_processor.extract_youtube_urls(content)
                if file_youtube_urls:
                    total_youtube_urls += len(file_youtube_urls)
                    print(f"   📺 {file_path.name}")
                    print(f"      → Found {len(file_youtube_urls)} YouTube URL(s)")
                
                # Split into chunks
                chunks = self.text_splitter.split_text(content)
                
                for idx, chunk in enumerate(chunks):
                    # Extract URLs from chunk
                    youtube_urls = self.doc_processor.extract_youtube_urls(chunk)
                    other_urls = self.doc_processor.extract_non_youtube_urls(chunk)
                    total_urls = len(youtube_urls) + len(other_urls)
                    
                    # Classify content
                    content_type = self.doc_processor.classify_content_type(
                        chunk, len(youtube_urls), len(other_urls)
                    )
                    
                    # Create metadata - STORE FULL FILENAME WITH EXTENSION
                    full_filename = file_path.name  # This includes .docx extension
                    
                    metadata = {
                        'section_type': section_type,
                        'section_name': section_name,
                        'session_number': session_num if session_num else -1,
                        'file_name': full_filename,  # NOW INCLUDES .docx
                        'original_file_path': str(file_path),
                        'chunk_index': idx,
                        'has_urls': total_urls > 0,
                        'youtube_url_count': len(youtube_urls),
                        'other_url_count': len(other_urls),
                        'total_url_count': total_urls,
                        'youtube_urls': '|'.join(youtube_urls[:5]),  # Store up to 5
                        'other_urls': '|'.join(other_urls[:3]),
                        'content_type': content_type,
                        'source': str(file_path)
                    }
                    
                    # Enhanced content with proper section labeling
                    if section_type == 'session' and session_num:
                        enhanced_content = f"Session {session_num}: {section_name}\nFile: {full_filename}\n\n{chunk}"
                    elif section_type == '100bm_community':
                        enhanced_content = f"100 BM Community: {full_filename}\n\n{chunk}"
                    elif section_type == 'lep_revision':
                        enhanced_content = f"LEP Revision: {full_filename}\n\n{chunk}"
                    else:
                        enhanced_content = f"Batch Schedule: {full_filename}\n\n{chunk}"
                    
                    doc = Document(
                        page_content=enhanced_content,
                        metadata=metadata
                    )
                    processed_docs.append(doc)
                
                print(f"   ✅ Processed: {file_path.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                print(f"   ❌ Error processing {file_path.name}: {str(e)}")
                continue
        
        print(f"\n✅ Summary:")
        print(f"   • Total chunks created: {len(processed_docs)}")
        print(f"   • Total YouTube URLs found: {total_youtube_urls}")
        
        return processed_docs
    
    def _create_url_mapping(self, documents: List[Document]):
        """Create JSON mapping file: filename → YouTube URLs"""
        print(f"\n📋 Creating filename → YouTube URL mapping...")
        
        # Build mapping: filename → set of URLs
        url_mapping = {}
        
        for doc in documents:
            filename = doc.metadata.get('file_name', '')
            urls_str = doc.metadata.get('youtube_urls', '')
            
            if filename and urls_str and urls_str.strip():
                # Split pipe-separated URLs
                urls = [u.strip() for u in urls_str.split('|') if u.strip()]
                
                if urls:
                    if filename not in url_mapping:
                        url_mapping[filename] = set()
                    url_mapping[filename].update(urls)
        
        # Convert sets to lists for JSON serialization
        url_mapping_json = {
            filename: list(urls) 
            for filename, urls in url_mapping.items()
        }
        
        # Save to JSON file
        output_file = './url_mapping.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(url_mapping_json, f, indent=2, ensure_ascii=False)
        
        print(f"✅ URL mapping saved to: {output_file}")
        print(f"   Total files with URLs: {len(url_mapping_json)}")
        
        # Show sample
        print(f"\n📊 Sample mappings:")
        for filename, urls in list(url_mapping_json.items())[:3]:
            print(f"   • {filename}")
            for url in urls:
                print(f"     → {url}")
        
        return url_mapping_json
    
    def create_vectorstore(self, folder_path: str):
        """Create vector store from folder structure"""
        documents = self.process_folder_structure(folder_path)
        
        if not documents:
            raise ValueError("No documents to create vector store")
        
        print(f"\n🚀 Creating vector store with {len(documents)} chunks...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"✅ Vector store created and persisted to {self.persist_directory}")
        
        # AUTOMATICALLY CREATE URL MAPPING FILE
        self._create_url_mapping(documents)
        
        return self.vectorstore
    
    def load_vectorstore(self):
        """Load existing vector store"""
        print(f"📂 Loading vector store from {self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        
        print("✅ Vector store loaded successfully")
        return self.vectorstore
    
    def query(
        self,
        query: str,
        k: int = 5,
        filter_section_type: Optional[str] = None,
        filter_session: Optional[int] = None,
        filter_content_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query with filters"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        where_filter = {}
        if filter_section_type:
            where_filter['section_type'] = filter_section_type
        if filter_session is not None:
            where_filter['session_number'] = filter_session
        if filter_content_type:
            where_filter['content_type'] = filter_content_type
        
        if where_filter:
            results = self.vectorstore.similarity_search(query, k=k, filter=where_filter)
        else:
            results = self.vectorstore.similarity_search(query, k=k)
        
        formatted_results = []
        for doc in results:
            youtube_urls = doc.metadata.get('youtube_urls', '').split('|') if doc.metadata.get('youtube_urls') else []
            other_urls = doc.metadata.get('other_urls', '').split('|') if doc.metadata.get('youtube_urls') else []
            
            result = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'section_type': doc.metadata.get('section_type', 'unknown'),
                'section_name': doc.metadata.get('section_name', 'Unknown'),
                'session_number': doc.metadata.get('session_number', -1),
                'file_name': doc.metadata.get('file_name', 'Unknown'),
                'youtube_urls': [url for url in youtube_urls if url],
                'other_urls': [url for url in other_urls if url],
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def get_100bm_community_content(self, k: int = 10) -> List[Dict[str, Any]]:
        """Get 100 BM Community content ONLY"""
        return self.query("100 BM Community", k=k, filter_section_type='100bm_community')
    
    def get_session_content(self, session_number: int, k: int = 15) -> List[Dict[str, Any]]:
        """Get session content ONLY"""
        return self.query(f"Session {session_number}", k=k, filter_session=session_number)


# For backward compatibility
EnhancedVectorStore = FolderVectorStore


def main():
    """Example usage"""
    print("="*80)
    print("🎓 100 BM Community Vector Store - WITH AUTO URL MAPPING")
    print("="*80)
    
    vector_store = FolderVectorStore()
    
    # Path to your folder structure
    folder_path = "./lms_content"
    
    if Path("./chroma_db").exists():
        print("\n⚠️  Existing vector store found.")
        print("❌ Delete ./chroma_db to rebuild with URL mapping!")
        return
    else:
        print("\n🆕 Creating new vector store with URL mapping...")
        if not Path(folder_path).exists():
            print(f"❌ Folder not found: {folder_path}")
            return
        vector_store.create_vectorstore(folder_path)
    
    print("\n" + "="*80)
    print("✅ Complete! Files created:")
    print("   1. ./chroma_db/ - Vector database")
    print("   2. ./url_mapping.json - Filename → URL mapping")
    print("="*80)


if __name__ == "__main__":
    main()