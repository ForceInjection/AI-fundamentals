source "https://rubygems.org"

gem "github-pages", group: :jekyll_plugins
gem "minima"

# Windows 和 JRuby 相关依赖
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# 性能优化，用于监视文件变动
gem "wdm", "~> 0.1", :platforms => [:mingw, :x64_mingw, :mswin]

# 锁定 http_parser.rb 版本，避免 JRuby 构建问题
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]
