# Makefile for MapReduce Framework
# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -pthread -O2
AR = ar
ARFLAGS = rcs

# Target library name
TARGET = libMapReduceFramework.a

# Source files
SOURCES = MapReduceFramework.cpp
HEADERS = MapReduceFramework.h MapReduceClient.h

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build the static library
$(TARGET): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install (if needed)
install: $(TARGET)
	@echo "Library built successfully: $(TARGET)"

# Rebuild everything
rebuild: clean all

# Show help
help:
	@echo "Available targets:"
	@echo "  all      - Build the library (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  rebuild  - Clean and build"
	@echo "  install  - Build and show completion message"
	@echo "  help     - Show this help"

# Declare phony targets
.PHONY: all clean install rebuild help