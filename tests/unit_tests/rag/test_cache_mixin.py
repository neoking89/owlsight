import pytest
import tempfile
from pathlib import Path
import shutil

from owlsight.rag.custom_classes import CacheMixin, _process_full_cache_path


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    temp_dir = tempfile.mkdtemp()
    cache_dir = Path(temp_dir) / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    yield cache_dir
    
    # Cleanup after tests
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_cache_mixin(temp_cache_dir):
    """Create a basic CacheMixin instance with a simple suffix."""
    return CacheMixin(
        cache_dir=str(temp_cache_dir),
        cache_dir_suffix="test_suffix"
    )


@pytest.fixture
def test_data():
    """Sample data for testing cache operations."""
    return {
        "key1": "value1",
        "key2": [1, 2, 3],
        "key3": {"nested": "data"}
    }


class TestCacheMixin:
    """Tests for the CacheMixin class."""
    
    def test_init_with_valid_params(self, temp_cache_dir):
        """Test initialization with valid parameters."""
        mixin = CacheMixin(
            cache_dir=str(temp_cache_dir),
            cache_dir_suffix="test_suffix"
        )
        
        assert mixin.cache_dir == temp_cache_dir
        assert mixin.cache_dir_suffix == "test_suffix"
        assert temp_cache_dir.exists()
    
    def test_init_without_cache_dir(self):
        """Test initialization without cache_dir."""
        mixin = CacheMixin()
        
        assert mixin.cache_dir is None
        assert mixin.cache_dir_suffix is None
    
    def test_init_with_cache_dir_but_no_suffix(self, temp_cache_dir):
        """Test initialization with cache_dir but no cache_dir_suffix."""
        with pytest.raises(ValueError, match="cache_dir_suffix must be provided"):
            CacheMixin(cache_dir=str(temp_cache_dir))
    
    def test_get_suffix_filename(self, simple_cache_mixin):
        """Test get_suffix_filename method."""
        assert simple_cache_mixin.get_suffix_filename() == "test_suffix"
    
    def test_get_suffix_filename_when_none(self):
        """Test get_suffix_filename method when suffix is None."""
        mixin = CacheMixin()
        assert mixin.get_suffix_filename() == ""
    
    def test_get_full_cache_path(self, simple_cache_mixin, temp_cache_dir):
        """Test get_full_cache_path method."""
        cache_path = simple_cache_mixin.get_full_cache_path()
        
        assert cache_path == temp_cache_dir / "test_suffix.pkl"
        assert cache_path.parent == temp_cache_dir
    
    def test_get_full_cache_path_without_cache_dir(self):
        """Test get_full_cache_path method when cache_dir is None."""
        mixin = CacheMixin()
        
        with pytest.raises(ValueError, match="Cache directory not provided"):
            mixin.get_full_cache_path()
    
    def test_save_and_load_data(self, simple_cache_mixin, test_data):
        """Test saving and loading data."""
        # Save data
        simple_cache_mixin.save_data(test_data)
        
        # Verify file exists
        cache_path = simple_cache_mixin.get_full_cache_path()
        assert cache_path.exists()
        
        # Load data
        loaded_data = simple_cache_mixin.load_data()
        
        # Verify data integrity
        assert loaded_data == test_data
        assert loaded_data["key1"] == "value1"
        assert loaded_data["key2"] == [1, 2, 3]
        assert loaded_data["key3"]["nested"] == "data"
    
    def test_load_data_when_file_not_exists(self, simple_cache_mixin):
        """Test loading data when cache file doesn't exist."""
        # Make sure file doesn't exist
        cache_path = simple_cache_mixin.get_full_cache_path()
        if cache_path.exists():
            cache_path.unlink()
        
        # Load data should return None
        assert simple_cache_mixin.load_data() is None
    
    def test_load_data_without_cache_dir(self):
        """Test loading data when cache_dir is None."""
        mixin = CacheMixin()
        
        # Should return None without raising an exception
        assert mixin.load_data() is None
    
    def test_save_data_without_cache_dir(self, test_data):
        """Test saving data when cache_dir is None."""
        mixin = CacheMixin()
        
        # Should not raise an exception
        mixin.save_data(test_data)
    
    def test_long_filename(self, temp_cache_dir, test_data):
        """Test handling of very long filenames (over 300 chars)."""
        # Create a very long suffix (over 300 chars)
        long_suffix = "a" * 310
        
        # Create a CacheMixin with the long suffix
        mixin = CacheMixin(
            cache_dir=str(temp_cache_dir),
            cache_dir_suffix=long_suffix
        )
        
        # Save data
        mixin.save_data(test_data)
        
        # Verify file exists with shortened name
        cache_path = mixin.get_full_cache_path()
        assert cache_path.exists()
        
        # Load data
        loaded_data = mixin.load_data()
        
        # Verify data integrity
        assert loaded_data == test_data
        
        # Verify filename was shortened
        filename = cache_path.name
        assert len(filename) < 300
        
        # The shortened filename should contain a hash part
        assert "_" in filename
    
    def test_extremely_long_filename(self, temp_cache_dir, test_data):
        """Test handling of extremely long filenames."""
        # Create an extremely long suffix (1000 chars)
        long_suffix = "b" * 1000
        
        # Create a CacheMixin with the long suffix
        mixin = CacheMixin(
            cache_dir=str(temp_cache_dir),
            cache_dir_suffix=long_suffix
        )
        
        # Save data
        mixin.save_data(test_data)
        
        # Load data
        loaded_data = mixin.load_data()
        
        # Verify data integrity
        assert loaded_data == test_data
        
        # Verify path handling
        cache_path = mixin.get_full_cache_path()
        assert len(str(cache_path)) <= 260  # Windows path length limit
    
    def test_process_full_cache_path_function(self, temp_cache_dir):
        """Test the internal _process_full_cache_path function directly."""
        # Normal case
        normal_suffix = "normal_suffix"
        normal_path = _process_full_cache_path(temp_cache_dir, normal_suffix)
        assert normal_path == temp_cache_dir / f"{normal_suffix}.pkl"
        
        # Long filename case
        long_suffix = "c" * 500
        long_path = _process_full_cache_path(temp_cache_dir, long_suffix)
        
        # Verify the path was shortened
        assert len(str(long_path)) < len(str(temp_cache_dir / f"{long_suffix}.pkl"))
        assert "_" in long_path.name  # Should contain the hash separator
        
        # The path should still have the correct extension
        assert long_path.suffix == ".pkl"


if __name__ == "__main__":
    pytest.main(["-vv", "-s", __file__])
