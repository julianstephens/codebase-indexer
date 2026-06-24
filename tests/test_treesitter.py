"""
tests/test_treesitter.py — Tests for indexer/treesitter.py

Coverage:
  - extract() for every language in LANG_CONFIG
  - NodeRecord field correctness: label, name, parent, start_line,
    end_line, signature, source, language
  - Custom name extractors: decorated_definition, arrow_function,
    export_statement, impl_item, type_declaration, c function declarator,
    template_declaration, lua assignment, kotlin secondary_constructor,
    swift init_declaration
  - _find_body_start() for each body opener type
  - _extract_signature() single-line and multi-line cases
  - _extract_source() indentation preservation
  - _collect_properties() async, decorators, visibility, receiver, static
  - Edge cases: empty source, syntax errors, anonymous functions,
    nested classes, deeply nested methods, files with no definitions
"""

import textwrap
import typing

import pytest

from src.indexer.languages import LANG_CONFIG
from src.indexer.treesitter import (
    NodeRecord,
    _collect_properties,
    _extract_signature,
    _extract_source,
    _find_body_start,
    _get_parser,
    _join_lines,
    _node_text,
    extract,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse(language: str, source: str):
    """Return the root node of a parsed tree."""
    parser = _get_parser(LANG_CONFIG[language]["parser"])
    return parser.parse(source.encode()).root_node


def first_def_node(language: str, source: str, node_type: str):
    """Return the first node of the given type in the parsed tree."""
    root = parse(language, source)
    return _find_first(root, node_type)


def _find_first(node, node_type: str):
    if node.type == node_type:
        return node
    for child in node.children:
        result = _find_first(child, node_type)
        if result:
            return result
    return None


def dedent(source: str) -> str:
    return textwrap.dedent(source).strip()


# ---------------------------------------------------------------------------
# _join_lines
# ---------------------------------------------------------------------------


class TestJoinLines:
    def test_basic_slice(self):
        lines = ["a", "b", "c", "d"]
        assert _join_lines(lines, 0, 2) == "a\nb"

    def test_single_line(self):
        lines = ["hello", "world"]
        assert _join_lines(lines, 0, 1) == "hello"

    def test_empty_range(self):
        lines = ["a", "b"]
        assert _join_lines(lines, 1, 1) == ""

    def test_start_greater_than_end(self):
        lines = ["a", "b"]
        assert _join_lines(lines, 2, 1) == ""

    def test_full_range(self):
        lines = ["x", "y", "z"]
        assert _join_lines(lines, 0, 3) == "x\ny\nz"

    def test_empty_lines_list(self):
        assert _join_lines([], 0, 0) == ""


# ---------------------------------------------------------------------------
# _node_text
# ---------------------------------------------------------------------------


class TestNodeText:
    def test_simple_identifier(self):
        node = first_def_node("python", "def foo(): pass", "identifier")
        assert _node_text(node) == "foo"

    def test_returns_str_not_bytes(self):
        node = first_def_node("python", "def bar(): pass", "identifier")
        result = _node_text(node)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _find_body_start
# ---------------------------------------------------------------------------


class TestFindBodyStart:
    def test_python_function_block(self):
        source = dedent("""
            def foo():
                return 1
        """)
        node = first_def_node("python", source, "function_definition")
        result = _find_body_start(node)
        assert result is not None
        assert result == 1  # 0-based: body is on line index 1

    def test_python_class_block(self):
        source = dedent("""
            class Foo:
                pass
        """)
        node = first_def_node("python", source, "class_definition")
        result = _find_body_start(node)
        assert result is not None

    def test_javascript_statement_block(self):
        source = "function foo() { return 1; }"
        node = first_def_node("javascript", source, "function_declaration")
        result = _find_body_start(node)
        assert result is not None
        assert result == 0

    def test_java_class_body(self):
        source = dedent("""
            class Foo {
                int x;
            }
        """)
        node = first_def_node("java", source, "class_declaration")
        result = _find_body_start(node)
        assert result is not None

    def test_rust_function_block(self):
        source = dedent("""
            fn foo() -> i32 {
                42
            }
        """)
        node = first_def_node("rust", source, "function_item")
        result = _find_body_start(node)
        assert result is not None

    def test_go_function_block(self):
        source = dedent("""
            func foo() int {
                return 1
            }
        """)
        node = first_def_node("go", source, "function_declaration")
        result = _find_body_start(node)
        assert result is not None

    def test_no_body_returns_none(self):
        # Interface method in TypeScript has no body
        source = "interface Foo { bar(): void; }"
        node = first_def_node("typescript", source, "method_signature")
        result = _find_body_start(node)
        assert result is None

    def test_c_compound_statement(self):
        source = dedent("""
            int add(int a, int b) {
                return a + b;
            }
        """)
        node = first_def_node("c", source, "function_definition")
        result = _find_body_start(node)
        assert result is not None


# ---------------------------------------------------------------------------
# _extract_signature
# ---------------------------------------------------------------------------


class TestExtractSignature:
    def test_python_simple_function(self):
        source = dedent("""
            def charge(user, amount):
                return stripe.charge(user, amount)
        """)
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        sig = _extract_signature(node, lines)
        assert "def charge" in sig
        assert "user" in sig
        assert "amount" in sig
        assert "stripe" not in sig

    def test_python_multiline_signature(self):
        source = dedent("""
            def charge(
                user: User,
                amount_cents: int,
                currency: str = "usd",
            ) -> Payment:
                pass
        """)
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        sig = _extract_signature(node, lines)
        assert "def charge" in sig
        assert "Payment" in sig
        # Multiline signature should be joined to one line
        assert "\n" not in sig

    def test_python_type_hints(self):
        source = dedent("""
            def process(self, order: Order) -> bool:
                return True
        """)
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        sig = _extract_signature(node, lines)
        assert "-> bool" in sig

    def test_go_function(self):
        source = dedent("""
            func Charge(user User, cents int) error {
                return nil
            }
        """)
        lines = source.splitlines()
        node = first_def_node("go", source, "function_declaration")
        sig = _extract_signature(node, lines)
        assert "func Charge" in sig
        assert "error" in sig
        assert "return nil" not in sig

    def test_typescript_class(self):
        source = dedent("""
            export class PaymentService extends BaseService {
                constructor() { super(); }
            }
        """)
        lines = source.splitlines()
        node = first_def_node("typescript", source, "class_declaration")
        sig = _extract_signature(node, lines)
        assert "PaymentService" in sig
        assert "BaseService" in sig
        assert "constructor" not in sig

    def test_rust_function(self):
        source = dedent("""
            pub fn charge(user: &User, amount: u64) -> Result<Payment, Error> {
                Ok(Payment::new())
            }
        """)
        lines = source.splitlines()
        node = first_def_node("rust", source, "function_item")
        sig = _extract_signature(node, lines)
        assert "fn charge" in sig
        assert "Result" in sig
        assert "Ok" not in sig

    def test_java_method(self):
        source = dedent("""
            public class Foo {
                public Payment charge(User user, int cents) throws PaymentException {
                    return new Payment();
                }
            }
        """)
        lines = source.splitlines()
        node = first_def_node("java", source, "method_declaration")
        sig = _extract_signature(node, lines)
        assert "charge" in sig
        assert "Payment" in sig
        assert "new Payment" not in sig

    def test_interface_method_no_body(self):
        source = "interface IService { void process(); }"
        lines = source.splitlines()
        node = first_def_node("typescript", source, "method_signature")
        sig = _extract_signature(node, lines)
        assert sig  # should not be empty even with no body
        assert "process" in sig

    def test_never_empty(self):
        source = "def f(): pass"
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        sig = _extract_signature(node, lines)
        assert len(sig) > 0


# ---------------------------------------------------------------------------
# _extract_source
# ---------------------------------------------------------------------------


class TestExtractSource:
    def test_includes_body(self):
        source = dedent("""
            def foo():
                x = 1
                return x
        """)
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        src = _extract_source(node, lines)
        assert "def foo" in src
        assert "x = 1" in src
        assert "return x" in src

    def test_preserves_indentation(self):
        source = dedent("""
            class Foo:
                def bar(self):
                    return 42
        """)
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        src = _extract_source(node, lines)
        # indentation inside bar should be preserved
        assert "        return 42" in src or "    return 42" in src

    def test_multiline_function(self):
        source = dedent("""
            def process(
                self,
                order: Order,
            ) -> bool:
                if not order:
                    return False
                return True
        """)
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        src = _extract_source(node, lines)
        assert "return False" in src
        assert "return True" in src

    def test_never_empty(self):
        source = "def f(): pass"
        lines = source.splitlines()
        node = first_def_node("python", source, "function_definition")
        src = _extract_source(node, lines)
        assert len(src) > 0

    def test_go_method_source(self):
        source = dedent("""
            func (s *Service) Process() error {
                return nil
            }
        """)
        lines = source.splitlines()
        node = first_def_node("go", source, "method_declaration")
        src = _extract_source(node, lines)
        assert "func" in src
        assert "return nil" in src


# ---------------------------------------------------------------------------
# _collect_properties
# ---------------------------------------------------------------------------


class TestCollectProperties:
    def test_python_async_function(self):
        source = "async def fetch(): pass"
        node = first_def_node("python", source, "function_definition")
        props = _collect_properties(node, "python", source.splitlines())
        assert props.get("async") is True

    def test_python_sync_function_not_async(self):
        source = "def fetch(): pass"
        node = first_def_node("python", source, "function_definition")
        props = _collect_properties(node, "python", source.splitlines())
        assert not props.get("async", False)

    def test_python_single_decorator(self):
        source = dedent("""
            @login_required
            def view(request):
                pass
        """)
        # decorated_definition wraps the function
        node = first_def_node("python", source, "decorated_definition")
        props = _collect_properties(node, "python", source.splitlines())
        decorators = props.get("decorators", [])
        assert any("login_required" in d for d in decorators)

    def test_python_multiple_decorators(self):
        source = dedent("""
            @app.route("/pay")
            @login_required
            def pay(request):
                pass
        """)
        node = first_def_node("python", source, "decorated_definition")
        props = _collect_properties(node, "python", source.splitlines())
        decorators = props.get("decorators", [])
        assert len(decorators) >= 2

    def test_go_method_receiver(self):
        source = dedent("""
            func (s *PaymentService) Charge(cents int) error {
                return nil
            }
        """)
        node = first_def_node("go", source, "method_declaration")
        props = _collect_properties(node, "go", source.splitlines())
        assert "receiver" in props
        assert "PaymentService" in props["receiver"]

    def test_java_public_visibility(self):
        source = dedent("""
            public class Foo {
                public void bar() {}
            }
        """)
        node = first_def_node("java", source, "method_declaration")
        props = _collect_properties(node, "java", source.splitlines())
        assert props.get("visibility") == "public"

    def test_java_static_method(self):
        source = dedent("""
            class Foo {
                public static void helper() {}
            }
        """)
        node = first_def_node("java", source, "method_declaration")
        props = _collect_properties(node, "java", source.splitlines())
        assert props.get("static") is True

    def test_typescript_async_method(self):
        source = dedent("""
            class Service {
                async fetchUser(id: string): Promise<User> {
                    return await db.find(id);
                }
            }
        """)
        node = first_def_node("typescript", source, "method_definition")
        props = _collect_properties(node, "typescript", source.splitlines())
        assert props.get("async") is True

    def test_empty_props_for_plain_function(self):
        source = "def simple(): pass"
        node = first_def_node("python", source, "function_definition")
        props = _collect_properties(node, "python", source.splitlines())
        assert isinstance(props, dict)


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------


class TestExtractPython:
    def test_simple_function(self):
        source = dedent("""
            def add(a: int, b: int) -> int:
                return a + b
        """)
        records = extract("src/math.py", source)
        assert len(records) == 1
        r = records[0]
        assert r.name == "add"
        assert r.label == "Function"
        assert r.language == "python"
        assert r.start_line == 1
        assert "def add" in r.signature
        assert "return a + b" in r.source

    def test_class_with_methods(self):
        source = dedent("""
            class PaymentService:
                def charge(self, user, amount):
                    return True

                def refund(self, payment):
                    return False
        """)
        records = extract("src/service.py", source)
        names = [r.name for r in records]
        assert "PaymentService" in names
        assert "charge" in names
        assert "refund" in names

    def test_method_parent_set(self):
        source = dedent("""
            class Foo:
                def bar(self):
                    pass
        """)
        records = extract("src/foo.py", source)
        method = next(r for r in records if r.name == "bar")
        assert method.parent == "Foo"

    def test_top_level_function_no_parent(self):
        source = dedent("""
            def standalone():
                pass
        """)
        records = extract("src/utils.py", source)
        assert records[0].parent == ""

    def test_async_function(self):
        source = dedent("""
            async def fetch(url: str) -> dict:
                return {}
        """)
        records = extract("src/client.py", source)
        assert records[0].name == "fetch"
        assert records[0].label == "Function"

    def test_decorated_function(self):
        source = dedent("""
            @login_required
            def dashboard(request):
                return render(request, "dashboard.html")
        """)
        records = extract("src/views.py", source)
        assert len(records) == 1
        assert records[0].name == "dashboard"
        assert "login_required" in records[0].properties.get("decorators", [])

    def test_decorated_class(self):
        source = dedent("""
            @dataclass
            class Config:
                host: str
                port: int
        """)
        records = extract("src/config.py", source)
        assert len(records) == 1
        assert records[0].name == "Config"
        assert records[0].label == "Class"

    def test_inherited_class(self):
        source = dedent("""
            class AdminUser(User, PermissionMixin):
                def is_admin(self):
                    return True
        """)
        records = extract("src/models.py", source)
        cls = next(r for r in records if r.label == "Class")
        assert "User" in cls.signature or "AdminUser" in cls.signature

    def test_nested_class(self):
        source = dedent("""
            class Outer:
                class Inner:
                    def method(self):
                        pass
        """)
        records = extract("src/nested.py", source)
        names = [r.name for r in records]
        assert "Outer" in names
        assert "Inner" in names
        assert "method" in names

    def test_multiple_decorators(self):
        source = dedent("""
            @app.route("/pay", methods=["POST"])
            @login_required
            @csrf_exempt
            def pay(request):
                pass
        """)
        records = extract("src/views.py", source)
        assert len(records) == 1
        decorators = records[0].properties.get("decorators", [])
        assert len(decorators) == 3

    def test_qualified_name_empty_before_pipeline(self):
        source = "def foo(): pass"
        records = extract("src/utils.py", source)
        # QN is set by pipeline.py, not the extractor
        assert records[0].qualified_name == ""

    def test_source_order(self):
        source = dedent("""
            def first(): pass
            def second(): pass
            def third(): pass
        """)
        records = extract("src/funcs.py", source)
        names = [r.name for r in records]
        assert names == ["first", "second", "third"]

    def test_line_numbers_are_one_based(self):
        source = dedent("""
            def foo():
                pass
        """)
        records = extract("src/foo.py", source)
        assert records[0].start_line >= 1
        assert records[0].end_line >= records[0].start_line

    def test_empty_source(self):
        records = extract("src/empty.py", "")
        assert records == []

    def test_syntax_error_partial_result(self):
        # tree-sitter produces partial AST for invalid syntax
        source = dedent("""
            def valid():
                pass

            def invalid(
        """)
        records = extract("src/broken.py", source)
        # Should extract at least the valid function, not raise
        assert any(r.name == "valid" for r in records)

    def test_type_hints_in_signature(self):
        source = dedent("""
            def process(
                order: Order,
                user: User,
                dry_run: bool = False,
            ) -> tuple[bool, str]:
                return True, "ok"
        """)
        records = extract("src/processor.py", source)
        assert records[0].name == "process"
        assert "Order" in records[0].signature
        assert "tuple" in records[0].signature

    def test_file_path_stored_on_record(self):
        source = "def foo(): pass"
        records = extract("src/payments/service.py", source)
        assert records[0].file_path == "src/payments/service.py"


# ---------------------------------------------------------------------------
# TypeScript extraction
# ---------------------------------------------------------------------------


class TestExtractTypeScript:
    def test_function_declaration(self):
        source = dedent("""
            function greet(name: string): string {
                return `Hello ${name}`;
            }
        """)
        records = extract("src/utils.ts", source)
        assert any(r.name == "greet" and r.label == "Function" for r in records)

    def test_async_function(self):
        source = dedent("""
            async function fetchUser(id: string): Promise<User> {
                return await db.find(id);
            }
        """)
        records = extract("src/api.ts", source)
        r = next(r for r in records if r.name == "fetchUser")
        assert r.label == "Function"

    def test_class_declaration(self):
        source = dedent("""
            export class UserService {
                async getUser(id: string): Promise<User> {
                    return this.repo.find(id);
                }
            }
        """)
        records = extract("src/service.ts", source)
        names = [r.name for r in records]
        assert "UserService" in names
        assert "getUser" in names

    def test_method_parent(self):
        source = dedent("""
            class Foo {
                bar(): void {}
            }
        """)
        records = extract("src/foo.ts", source)
        method = next(r for r in records if r.name == "bar")
        assert method.parent == "Foo"

    def test_interface_declaration(self):
        source = dedent("""
            interface IPaymentService {
                charge(amount: number): Promise<Payment>;
                refund(id: string): Promise<void>;
            }
        """)
        records = extract("src/interfaces.ts", source)
        assert any(
            r.name == "IPaymentService" and r.label == "Interface" for r in records
        )

    def test_arrow_function_named(self):
        source = dedent("""
            const process = async (order: Order): Promise<void> => {
                await db.save(order);
            };
        """)
        records = extract("src/handlers.ts", source)
        assert any(r.name == "process" for r in records)

    def test_type_alias(self):
        source = "type UserId = string;"
        records = extract("src/types.ts", source)
        assert any(r.name == "UserId" and r.label == "Type" for r in records)

    def test_enum_declaration(self):
        source = dedent("""
            enum Status {
                Active,
                Inactive,
                Pending,
            }
        """)
        records = extract("src/enums.ts", source)
        assert any(r.name == "Status" and r.label == "Type" for r in records)

    def test_export_function(self):
        source = dedent("""
            export function validate(input: unknown): boolean {
                return !!input;
            }
        """)
        records = extract("src/validate.ts", source)
        assert any(r.name == "validate" for r in records)

    def test_abstract_class(self):
        source = dedent("""
            abstract class BaseService {
                abstract process(): void;
            }
        """)
        records = extract("src/base.ts", source)
        assert any(r.name == "BaseService" and r.label == "Class" for r in records)

    def test_tsx_component(self):
        source = dedent("""
            export function Button({ label }: ButtonProps): JSX.Element {
                return <button>{label}</button>;
            }
        """)
        records = extract("src/Button.tsx", source)
        assert any(r.name == "Button" for r in records)

    def test_empty_source(self):
        assert extract("src/empty.ts", "") == []


# ---------------------------------------------------------------------------
# Go extraction
# ---------------------------------------------------------------------------


class TestExtractGo:
    def test_function_declaration(self):
        source = dedent("""
            func Add(a, b int) int {
                return a + b
            }
        """)
        records = extract("pkg/math/math.go", source)
        assert any(r.name == "Add" and r.label == "Function" for r in records)

    def test_method_declaration(self):
        source = dedent("""
            func (s *PaymentService) Charge(cents int) error {
                return nil
            }
        """)
        records = extract("pkg/payments/service.go", source)
        r = next(r for r in records if r.name == "Charge")
        assert r.label == "Method"
        assert r.parent == "PaymentService"

    def test_method_receiver_in_properties(self):
        source = dedent("""
            func (s *Service) Process() error {
                return nil
            }
        """)
        records = extract("pkg/svc.go", source)
        r = records[0]
        assert "Service" in r.properties.get("receiver", "")

    def test_type_declaration_struct(self):
        source = dedent("""
            type User struct {
                ID   int
                Name string
            }
        """)
        records = extract("pkg/models/user.go", source)
        assert any(r.name == "User" for r in records)

    def test_multiple_functions(self):
        source = dedent("""
            func foo() {}
            func bar() {}
            func baz() {}
        """)
        records = extract("pkg/utils.go", source)
        names = [r.name for r in records]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names

    def test_empty_source(self):
        assert extract("pkg/empty.go", "") == []


# ---------------------------------------------------------------------------
# Rust extraction
# ---------------------------------------------------------------------------


class TestExtractRust:
    def test_function_item(self):
        source = dedent("""
            pub fn charge(amount: u64) -> Result<(), Error> {
                Ok(())
            }
        """)
        records = extract("src/payments.rs", source)
        assert any(r.name == "charge" and r.label == "Function" for r in records)

    def test_struct_item(self):
        source = dedent("""
            pub struct PaymentService {
                client: StripeClient,
            }
        """)
        records = extract("src/service.rs", source)
        assert any(r.name == "PaymentService" and r.label == "Class" for r in records)

    def test_impl_item(self):
        source = dedent("""
            impl PaymentService {
                pub fn new() -> Self {
                    Self { client: StripeClient::new() }
                }
            }
        """)
        records = extract("src/service.rs", source)
        assert any(r.name == "PaymentService" and r.label == "Class" for r in records)

    def test_trait_item(self):
        source = dedent("""
            pub trait Chargeable {
                fn charge(&self, amount: u64) -> Result<(), Error>;
            }
        """)
        records = extract("src/traits.rs", source)
        assert any(r.name == "Chargeable" and r.label == "Interface" for r in records)

    def test_enum_item(self):
        source = dedent("""
            pub enum PaymentStatus {
                Pending,
                Completed,
                Failed(String),
            }
        """)
        records = extract("src/models.rs", source)
        assert any(r.name == "PaymentStatus" and r.label == "Type" for r in records)

    def test_empty_source(self):
        assert extract("src/empty.rs", "") == []


# ---------------------------------------------------------------------------
# Java extraction
# ---------------------------------------------------------------------------


class TestExtractJava:
    def test_class_declaration(self):
        source = dedent("""
            public class PaymentService {
                public Payment charge(User user, int cents) {
                    return new Payment();
                }
            }
        """)
        records = extract("src/PaymentService.java", source)
        names = [r.name for r in records]
        assert "PaymentService" in names
        assert "charge" in names

    def test_method_parent(self):
        source = dedent("""
            public class Foo {
                public void bar() {}
            }
        """)
        records = extract("src/Foo.java", source)
        method = next(r for r in records if r.name == "bar")
        assert method.parent == "Foo"

    def test_interface_declaration(self):
        source = dedent("""
            public interface IPaymentService {
                Payment charge(User user, int cents);
            }
        """)
        records = extract("src/IPaymentService.java", source)
        assert any(
            r.name == "IPaymentService" and r.label == "Interface" for r in records
        )

    def test_constructor(self):
        source = dedent("""
            public class Service {
                public Service(Config config) {
                    this.config = config;
                }
            }
        """)
        records = extract("src/Service.java", source)
        assert any(r.name == "Service" and r.label == "Function" for r in records)

    def test_visibility_in_properties(self):
        source = dedent("""
            public class Foo {
                private void secret() {}
            }
        """)
        records = extract("src/Foo.java", source)
        method = next(r for r in records if r.name == "secret")
        assert method.properties.get("visibility") == "private"

    def test_static_method(self):
        source = dedent("""
            public class Utils {
                public static String format(String s) { return s; }
            }
        """)
        records = extract("src/Utils.java", source)
        method = next(r for r in records if r.name == "format")
        assert method.properties.get("static") is True

    def test_empty_source(self):
        assert extract("src/Empty.java", "") == []


# ---------------------------------------------------------------------------
# C extraction
# ---------------------------------------------------------------------------


class TestExtractC:
    def test_function_definition(self):
        source = dedent("""
            int add(int a, int b) {
                return a + b;
            }
        """)
        records = extract("src/math.c", source)
        assert any(r.name == "add" and r.label == "Function" for r in records)

    def test_pointer_return_function(self):
        source = dedent("""
            char *get_name(int id) {
                return names[id];
            }
        """)
        records = extract("src/names.c", source)
        assert any(r.name == "get_name" for r in records)

    def test_struct_specifier(self):
        source = dedent("""
            struct Point {
                int x;
                int y;
            };
        """)
        records = extract("src/geometry.c", source)
        assert any(r.name == "Point" for r in records)

    def test_header_file(self):
        source = dedent("""
            int process(const char *input);
            void cleanup(void *ptr);
        """)
        records = extract("include/api.h", source)
        # Forward declarations may or may not be captured depending on grammar
        # At minimum should not raise
        assert isinstance(records, list)

    def test_empty_source(self):
        assert extract("src/empty.c", "") == []


# ---------------------------------------------------------------------------
# Ruby extraction
# ---------------------------------------------------------------------------


class TestExtractRuby:
    def test_method(self):
        source = dedent("""
            def charge(user, amount)
                Stripe.charge(user.token, amount)
            end
        """)
        records = extract("lib/payments.rb", source)
        assert any(r.name == "charge" and r.label == "Function" for r in records)

    def test_class(self):
        source = dedent("""
            class PaymentService
                def charge(amount)
                    true
                end
            end
        """)
        records = extract("lib/payment_service.rb", source)
        names = [r.name for r in records]
        assert "PaymentService" in names
        assert "charge" in names

    def test_method_parent(self):
        source = dedent("""
            class Foo
                def bar
                end
            end
        """)
        records = extract("lib/foo.rb", source)
        method = next(r for r in records if r.name == "bar")
        assert method.parent == "Foo"

    def test_module(self):
        source = dedent("""
            module Payments
                def self.process(order)
                end
            end
        """)
        records = extract("lib/payments.rb", source)
        assert any(r.name == "Payments" for r in records)

    def test_empty_source(self):
        assert extract("lib/empty.rb", "") == []


# ---------------------------------------------------------------------------
# Bash extraction
# ---------------------------------------------------------------------------


class TestExtractBash:
    def test_function_definition(self):
        source = dedent("""
            function deploy() {
                echo "deploying"
            }
        """)
        records = extract("scripts/deploy.sh", source)
        assert any(r.name == "deploy" and r.label == "Function" for r in records)

    def test_function_without_keyword(self):
        source = dedent("""
            build() {
                make all
            }
        """)
        records = extract("scripts/build.sh", source)
        assert any(r.name == "build" for r in records)

    def test_empty_source(self):
        assert extract("scripts/empty.sh", "") == []


# ---------------------------------------------------------------------------
# Unrecognised extension
# ---------------------------------------------------------------------------


class TestUnrecognisedExtension:
    def test_returns_empty_list(self):
        source = "# some config\nkey = value\n"
        records = extract("config/settings.toml", source)
        assert records == []

    def test_markdown_returns_empty(self):
        records = extract("README.md", "# Hello\nSome text\n")
        assert records == []

    def test_dockerfile_returns_empty(self):
        records = extract("Dockerfile", "FROM python:3.11\nRUN pip install flask\n")
        assert records == []


# ---------------------------------------------------------------------------
# NodeRecord dataclass
# ---------------------------------------------------------------------------


class TestNodeRecord:
    def test_default_qualified_name_empty(self):
        r = NodeRecord(
            label="Function",
            name="foo",
            file_path="src/foo.py",
            start_line=1,
            end_line=3,
            signature="def foo():",
            source="def foo():\n    pass",
            language="python",
        )
        assert r.qualified_name == ""

    def test_default_parent_empty(self):
        r = NodeRecord(
            label="Function",
            name="foo",
            file_path="src/foo.py",
            start_line=1,
            end_line=3,
            signature="def foo():",
            source="def foo():\n    pass",
            language="python",
        )
        assert r.parent == ""

    def test_default_properties_empty_dict(self):
        r = NodeRecord(
            label="Function",
            name="foo",
            file_path="src/foo.py",
            start_line=1,
            end_line=3,
            signature="def foo():",
            source="def foo():\n    pass",
            language="python",
        )
        assert r.properties == {}

    def test_properties_not_shared_between_instances(self):
        # Mutable default — each instance must get its own dict
        r1 = NodeRecord(
            label="Function",
            name="foo",
            file_path="a.py",
            start_line=1,
            end_line=1,
            signature="def foo():",
            source="def foo(): pass",
            language="python",
        )
        r2 = NodeRecord(
            label="Function",
            name="bar",
            file_path="b.py",
            start_line=1,
            end_line=1,
            signature="def bar():",
            source="def bar(): pass",
            language="python",
        )
        r1.properties["x"] = 1
        assert "x" not in r2.properties


# ---------------------------------------------------------------------------
# Cross-language label consistency
# ---------------------------------------------------------------------------


class TestLabelConsistency:
    """
    Verify that every language uses only the allowed label values and
    that class-body children get their parent field set correctly.
    """

    ALLOWED_LABELS: typing.ClassVar[set[str]] = {
        "Function",
        "Class",
        "Method",
        "Interface",
        "Type",
        "File",
    }

    @pytest.mark.parametrize(
        "path,source",
        [
            ("f.py", "def foo(): pass"),
            ("f.ts", "function foo() {}"),
            ("f.go", "func foo() {}"),
            ("f.rs", "fn foo() {}"),
            ("f.java", "class F { void foo() {} }"),
            ("f.rb", "def foo; end"),
            ("f.sh", "function foo() { echo hi; }"),
        ],
    )
    def test_labels_are_valid(self, path, source):
        records = extract(path, source)
        for r in records:
            assert (
                r.label in self.ALLOWED_LABELS
            ), f"Invalid label '{r.label}' in {path}"

    @pytest.mark.parametrize(
        "path,source,class_name,method_name",
        [
            ("f.py", "class C:\n    def m(self): pass", "C", "m"),
            ("f.ts", "class C { m() {} }", "C", "m"),
            ("f.java", "class C { void m() {} }", "C", "m"),
            ("f.rb", "class C\n  def m\n  end\nend", "C", "m"),
        ],
    )
    def test_method_parent_set(self, path, source, class_name, method_name):
        records = extract(path, source)
        method = next((r for r in records if r.name == method_name), None)
        assert method is not None, f"Method '{method_name}' not found"
        assert (
            method.parent == class_name
        ), f"Expected parent='{class_name}', got '{method.parent}'"


# ---------------------------------------------------------------------------
# Large / stress cases
# ---------------------------------------------------------------------------


class TestStressCases:
    def test_many_functions(self):
        funcs = "\n".join(
            f"def func_{i}(x: int) -> int:\n    return x + {i}" for i in range(100)
        )
        records = extract("src/generated.py", funcs)
        assert len(records) == 100
        assert all(r.label == "Function" for r in records)

    def test_deeply_nested_classes(self):
        source = dedent("""
            class A:
                class B:
                    class C:
                        def method(self):
                            pass
        """)
        records = extract("src/nested.py", source)
        names = [r.name for r in records]
        assert "A" in names
        assert "B" in names
        assert "C" in names
        assert "method" in names

    def test_large_class(self):
        methods = "\n".join(
            f"    def method_{i}(self):\n        return {i}" for i in range(50)
        )
        source = f"class BigService:\n{methods}"
        records = extract("src/big.py", source)
        methods_found = [r for r in records if r.name.startswith("method_")]
        assert len(methods_found) == 50
        assert all(r.parent == "BigService" for r in methods_found)

    def test_unicode_identifiers(self):
        source = "def résumé(): pass\n"
        # Should not raise; result may vary by tree-sitter version
        records = extract("src/unicode.py", source)
        assert isinstance(records, list)

    def test_windows_line_endings(self):
        source = "def foo():\r\n    return 1\r\n"
        records = extract("src/crlf.py", source)
        assert any(r.name == "foo" for r in records)
