# Set pandoc to PATH and check
Sys.setenv(RSTUDIO_PANDOC = "C:/Program Files/RStudio/resources/app/bin/quarto/bin")
Sys.setenv(PATH = paste("C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools", Sys.getenv("PATH"), sep = ";"))

# Try rmarkdown version
pv <- tryCatch(rmarkdown::pandoc_version(), error = function(e) NULL)
cat("pandoc version (tryCatch):", if (is.null(pv)) "FAILED" else as.character(pv), "\n")

# Direct call
pd_path <- "C:/Program Files/RStudio/resources/app/bin/quarto/bin/tools/pandoc.exe"
cat("Direct call to pandoc:\n")
system2(pd_path, "--version", stdout = TRUE, stderr = TRUE)
